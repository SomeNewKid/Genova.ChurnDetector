// File: Genova.ChurnDetector/Detector.cs
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Reflection;

namespace Genova.ChurnDetector;

public sealed class Detector
{
    // Artifact file names (embedded resources)
    internal const string ManifestResourceName = "manifest.json";
    internal const string CentroidsResourceName = "centroids.json";
    internal const string SeedBankResourceName = "seeds.json";

    private const float EPS = 1e-4f;   // tolerance for float comparisons

    // k-NN blend knobs (vote mixed with prototype similarity)
    private const int K_NEIGHBORS = 9;
    private const float ALPHA = 0.60f; // churnScore = ALPHA*protoSim + (1-ALPHA)*vote

    private readonly LocalHashEmbedder _embedder;

    // Prototypes grouped by label (centered space)
    private readonly float[][] _churnCentroids;
    private readonly float[][] _keepCentroids;
    private readonly float[][] _unsureCentroids;

    // Global mean vector for mean-centering (null = disabled)
    private readonly float[]? _globalMean;

    // Compact seed bank for k-NN vote (centered space)
    private readonly (float[] V, string Label)[] _seeds;

    private readonly float _churnMinSim;
    private readonly float _minMargin;

    // Narrow churn cues + negation guard
    public static readonly Regex ChurnCueRegex = new(
        @"(?ix)
      \b(
          cancel
        | terminate
        | unsubscribe
        | opt \s* out
        | stop \s+ (?:my|our|the) \s+ (?:subscription|service|plan|membership|auto \s* renew(?:al)?)
        | stop \s+ renew(?:al)?
        | close \s+ (?:my|our|the) \s+ account
        | end \s+ (?:my|our|the) \s+ (?:plan|account|subscription|service|membership)
      )\b",
        RegexOptions.Compiled | RegexOptions.CultureInvariant
    );

    public static readonly Regex NegationNearCueRegex = new(
        @"(?ix)
      \b(?:dont|don't|do \s* not|never|not|no \s* need \s* to)\b
      [^\w]{0,10}
      (?:cancel|terminate|unsubscribe|opt \s* out|stop|close|end)\b",
        RegexOptions.Compiled | RegexOptions.CultureInvariant
    );

    public Detector()
    {
        var asm = typeof(Detector).Assembly;
        var artifacts = ArtifactLoader.Load(asm);

        _embedder = new LocalHashEmbedder(artifacts.Manifest.vector_dim);

        _churnMinSim = artifacts.Manifest.thresholds.churn_min_similarity;
        _minMargin = artifacts.Manifest.thresholds.min_margin;

        // Load centroids (also fetch optional __global_mean)
        var by = artifacts.Centroids
            .GroupBy(c => c.label)
            .ToDictionary(g => g.Key, g => g.Select(c => c.vector).ToArray(), StringComparer.OrdinalIgnoreCase);

        bool has(string k) => by.ContainsKey(k) && by[k].Length > 0;
        if (!has("churn") || !has("keep") || !has("unsure"))
            throw new InvalidOperationException("Centroids must include labels: churn, keep, unsure.");

        _churnCentroids = by["churn"];
        _keepCentroids = by["keep"];
        _unsureCentroids = by["unsure"];
        _globalMean = by.TryGetValue("__global_mean", out var gm) ? gm.FirstOrDefault() : null;

        // Load compact seed bank (optional)
        _seeds = LoadSeeds(asm);
    }

    /// <summary>
    /// Returns ("churn" or "not_churn", confidence 0..1)
    /// </summary>
    public (string Label, float Confidence) GetSignal(string input)
    {
        var v = _embedder.Embed(input ?? string.Empty);
        v = ApplyMeanCenter(v, _globalMean); // center + L2-normalize if mean present

        // Max-over-prototypes per label (centered space)
        float churnProto = MaxSim(v, _churnCentroids);
        float keepSim = MaxSim(v, _keepCentroids);
        float unsureSim = MaxSim(v, _unsureCentroids);

        // k-NN vote over seed bank (centered space)
        float vote = ComputeChurnVote(v, K_NEIGHBORS);
        float churnScore = ALPHA * churnProto + (1 - ALPHA) * vote;

        // Determine top-2 and margin using prototype sims (ranking unchanged)
        Span<float> sims = stackalloc float[3] { churnProto, keepSim, unsureSim };
        var (topIdx, secondIdx, top, second) = VectorMath.Top2(sims);
        var margin = top - second;

        bool hasCue = ChurnCueRegex.IsMatch(input ?? string.Empty);
        bool negatesCue = NegationNearCueRegex.IsMatch(input ?? string.Empty);

        // Allow churn only if cue present and not negated;
        // threshold uses blended churnScore; rank still uses prototypes.
        if (hasCue && !negatesCue &&
            topIdx == 0 &&
            (churnScore + EPS) >= _churnMinSim &&
            (margin + EPS) >= _minMargin)
        {
            return ("churn", Math.Clamp(churnScore, 0f, 1f));
        }

        var bestNonChurn = Math.Max(keepSim, unsureSim);
        return ("not_churn", Math.Clamp(bestNonChurn, 0f, 1f));
    }

    // --- DEBUG HELPERS ---

    public (string[] Labels, float[] Similarities) GetRawSimilarities(string input)
    {
        var v = _embedder.Embed(input ?? string.Empty);
        v = ApplyMeanCenter(v, _globalMean);

        var churnProto = MaxSim(v, _churnCentroids);
        var keepSim = MaxSim(v, _keepCentroids);
        var unsureSim = MaxSim(v, _unsureCentroids);
        return (new[] { "churn", "keep", "unsure" }, new[] { churnProto, keepSim, unsureSim });
    }

    public string Explain(string input)
    {
        var v = _embedder.Embed(input ?? string.Empty);
        v = ApplyMeanCenter(v, _globalMean);

        float churnProto = MaxSim(v, _churnCentroids);
        float keepSim = MaxSim(v, _keepCentroids);
        float unsureSim = MaxSim(v, _unsureCentroids);
        float vote = ComputeChurnVote(v, K_NEIGHBORS);
        float churnScore = ALPHA * churnProto + (1 - ALPHA) * vote;

        Span<float> sims = stackalloc float[3] { churnProto, keepSim, unsureSim };
        var (topIdx, secondIdx, top, second) = VectorMath.Top2(sims);
        var margin = top - second;

        var topLabel = topIdx switch { 0 => "churn", 1 => "keep", _ => "unsure" };
        bool cue = ChurnCueRegex.IsMatch(input ?? string.Empty);
        bool thrOK = (churnScore + EPS) >= _churnMinSim;
        bool marOK = (topIdx == 0) ? ((margin + EPS) >= _minMargin) : false;

        string reason;
        if (!cue) reason = "blocked: cue=false";
        else if (topIdx != 0) reason = $"blocked: top='{topLabel}' (top={top:F2}) > churnProto({churnProto:F2})";
        else if (!thrOK) reason = $"blocked: blended threshold (score={churnScore:F2} < min={_churnMinSim:F2})";
        else if (!marOK) reason = $"blocked: margin (margin={margin:F2} < min={_minMargin:F2})";
        else reason = "churn: passed cue, threshold, and margin";

        return
          $"Explain: cue={(cue ? "true" : "false")}, " +
          $"proto={churnProto:F4}, vote={vote:F2}, blended={churnScore:F4}, " +
          $"keep={keepSim:F4}, unsure={unsureSim:F4}, top={topLabel}, margin={margin:F4}, " +
          $"bars(thr={_churnMinSim:F4}, minMargin={_minMargin:F4}, eps={EPS:G}) -> {reason}";
    }

    // --- Internals ---

    private static float[] ApplyMeanCenter(float[] v, float[]? mu)
    {
        if (mu is null) return v;
        var outv = (float[])v.Clone();
        double sum = 0;
        for (int i = 0; i < outv.Length; i++)
        {
            outv[i] -= mu[i];
            sum += (double)outv[i] * outv[i];
        }
        var norm = Math.Sqrt(sum);
        if (norm > 0)
        {
            var inv = (float)(1.0 / norm);
            for (int i = 0; i < outv.Length; i++) outv[i] *= inv;
        }
        return outv;
    }

    private static float MaxSim(ReadOnlySpan<float> v, float[][] centroids)
    {
        float best = float.NegativeInfinity;
        for (int i = 0; i < centroids.Length; i++)
        {
            var s = VectorMath.Dot(v, centroids[i]);
            if (s > best) best = s;
        }
        return best;
    }

    private (float[] V, string Label)[] LoadSeeds(Assembly asm)
    {
        try
        {
            var name = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith("." + SeedBankResourceName, StringComparison.OrdinalIgnoreCase));
            if (name is null) return Array.Empty<(float[] V, string Label)>();

            using var s = asm.GetManifestResourceStream(name);
            if (s is null) return Array.Empty<(float[] V, string Label)>();
            using var doc = JsonDocument.Parse(s);
            if (!doc.RootElement.TryGetProperty("seeds", out var arr)) return Array.Empty<(float[] V, string Label)>();

            var list = new List<(float[] V, string Label)>(arr.GetArrayLength());
            foreach (var e in arr.EnumerateArray())
            {
                var label = e.GetProperty("label").GetString() ?? "";
                var vecEl = e.GetProperty("vector");
                var vec = new float[vecEl.GetArrayLength()];
                int i = 0;
                foreach (var v in vecEl.EnumerateArray()) vec[i++] = v.GetSingle();
                list.Add((vec, label));
            }
            return list.ToArray();
        }
        catch
        {
            return Array.Empty<(float[] V, string Label)>();
        }
    }

    private float ComputeChurnVote(ReadOnlySpan<float> v, int k)
    {
        if (_seeds.Length == 0 || k <= 0) return 0f;

        // Compute similarities to all seeds; pick top-k
        var sims = new List<(float sim, string label)>(_seeds.Length);
        foreach (var s in _seeds)
        {
            sims.Add((VectorMath.Dot(v, s.V), s.Label));
        }
        sims.Sort((a, b) => b.sim.CompareTo(a.sim));
        int kk = Math.Min(k, sims.Count);

        int churnCount = 0;
        for (int i = 0; i < kk; i++)
            if (string.Equals(sims[i].label, "churn", StringComparison.OrdinalIgnoreCase)) churnCount++;

        return (float)churnCount / Math.Max(1, kk);
    }
}
