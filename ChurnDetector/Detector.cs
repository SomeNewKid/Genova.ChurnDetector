// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using Genova.Common.Attributes;

namespace Genova.ChurnDetector;

/// <summary>
/// Provides churn-signal detection over text using fixed embeddings, centroids, and an optional k-NN seed vote.
/// </summary>
[SuppressMessage(
    "StyleCop.CSharp.NamingRules",
    "SA1310:Field names should not contain underscore",
    Justification = "Readability of constants.")]
[SuppressMessage(
    "StyleCop.CSharp.SpacingRules",
    "SA1011:Closing square brackets should be spaced correctly",
    Justification = "Conflicting styling rules.")]
[CodeQuality(Public = true, Justification = "Intended for use by the Rusty Kane website.")]
public sealed partial class Detector
{
    /// <summary>
    /// Gets the manifest resource file name (embedded JSON) that defines model metadata and thresholds.
    /// </summary>
    internal const string ManifestResourceName = "manifest.json";

    /// <summary>
    /// Gets the centroids resource file name (embedded JSON) that holds label prototypes and the optional global mean.
    /// </summary>
    internal const string CentroidsResourceName = "centroids.json";

    /// <summary>
    /// Gets the seed bank resource file name (embedded JSON) used for the optional k-NN vote.
    /// </summary>
    internal const string SeedBankResourceName = "seeds.json";

    private const float EPS = 1e-4f; // tolerance for float comparisons

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

    /// <summary>
    /// Initializes a new instance of the <see cref="Detector"/> class by loading embedded artifacts.
    /// </summary>
    public Detector()
    {
        Assembly asm = typeof(Detector).Assembly;
        ModelArtifacts artifacts = ArtifactLoader.Load(asm);

        _embedder = new LocalHashEmbedder(artifacts.Manifest.VectorDim);

        _churnMinSim = artifacts.Manifest.Thresholds.ChurnMinSimilarity;
        _minMargin = artifacts.Manifest.Thresholds.MinMargin;

        // Load centroids (also fetch optional __global_mean).
        Dictionary<string, float[][]> by = artifacts.Centroids
            .GroupBy(c => c.Label)
            .ToDictionary(
                g => g.Key,
                g => g.Select(c => c.Vector).ToArray(),
                StringComparer.OrdinalIgnoreCase);

        bool HasLabel(string key)
        {
            return by.ContainsKey(key) && by[key].Length > 0;
        }

        if (!HasLabel("churn") || !HasLabel("keep") || !HasLabel("unsure"))
        {
            throw new InvalidOperationException("Centroids must include labels: churn, keep, unsure.");
        }

        _churnCentroids = by["churn"];
        _keepCentroids = by["keep"];
        _unsureCentroids = by["unsure"];
        float[][]? gm;
        _globalMean = by.TryGetValue("__global_mean", out gm) ? gm.FirstOrDefault() : null;

        // Load compact seed bank (optional).
        _seeds = LoadSeeds(asm);
    }

    /// <summary>
    /// Gets the compiled regular expression that detects churn cues (verbs with account-like objects).
    /// </summary>
    [GeneratedRegex(@"(?ix)
        \b(
          cancel
        | terminate
        | unsubscribe
        | opt \s* out
        | stop \s+ (?:my|our|the) \s+ (?:subscription|service|plan|membership|auto \s* renew(?:al)?)
        | stop \s+ renew(?:al)?
        | close \s+ (?:my|our|the) \s+ account
        | end \s+ (?:my|our|the) \s+ (?:plan|account|subscription|service|membership)
        )\b", RegexOptions.Compiled | RegexOptions.CultureInvariant)]
    public static partial Regex ChurnCueRegex();

    /// <summary>
    /// Gets the compiled regular expression that detects negation near a churn cue (e.g., "do not cancel").
    /// </summary>
    [GeneratedRegex(@"(?ix)
        \b(?:dont|don't|do \s* not|never|not|no \s* need \s* to)\b
        [^\w]{0,10}
        (?:cancel|terminate|unsubscribe|opt \s* out|stop|close|end)\b", RegexOptions.Compiled | RegexOptions.CultureInvariant)]
    public static partial Regex NegationNearCueRegex();

    /// <summary>
    /// Gets the churn signal for a single input.
    /// </summary>
    /// <param name="input">The user input text to score.</param>
    /// <returns>
    /// A tuple of the predicted label (<c>"churn"</c> or <c>"not_churn"</c>) and a confidence score in the range [0, 1].
    /// </returns>
    public (string Label, float Confidence) GetSignal(string input)
    {
        float[] v = _embedder.Embed(input ?? string.Empty);
        v = ApplyMeanCenter(v, _globalMean); // center + L2-normalize if mean present

        // Max-over-prototypes per label (centered space).
        float churnProto = MaxSim(v, _churnCentroids);
        float keepSim = MaxSim(v, _keepCentroids);
        float unsureSim = MaxSim(v, _unsureCentroids);

        // k-NN vote over seed bank (centered space).
        float vote = ComputeChurnVote(v, K_NEIGHBORS);
        float churnScore = (ALPHA * churnProto) + ((1 - ALPHA) * vote);

        // Determine top-2 and margin using prototype sims (ranking unchanged).
        Span<float> sims = [churnProto, keepSim, unsureSim];
        var result = VectorMath.Top2(sims);
        int topIdx = result.TopIdx;
        float top = result.Top;
        float second = result.Second;
        float margin = top - second;

        bool hasCue = ChurnCueRegex().IsMatch(input ?? string.Empty);
        bool negatesCue = NegationNearCueRegex().IsMatch(input ?? string.Empty);

        // Allow churn only if cue present and not negated;
        // threshold uses blended churnScore; rank still uses prototypes.
        if (hasCue && !negatesCue
            && topIdx == 0
            && (churnScore + EPS) >= _churnMinSim
            && (margin + EPS) >= _minMargin)
        {
            return ("churn", Math.Clamp(churnScore, 0f, 1f));
        }

        float bestNonChurn = Math.Max(keepSim, unsureSim);
        return ("not_churn", Math.Clamp(bestNonChurn, 0f, 1f));
    }

    /// <summary>
    /// Gets the raw prototype similarities (churn, keep, unsure) for debugging.
    /// </summary>
    /// <param name="input">The user input text to score.</param>
    /// <returns>An array of labels and a parallel array of prototype similarities.</returns>
    public (string[] Labels, float[] Similarities) GetRawSimilarities(string input)
    {
        float[] v = _embedder.Embed(input ?? string.Empty);
        v = ApplyMeanCenter(v, _globalMean);

        float churnProto = MaxSim(v, _churnCentroids);
        float keepSim = MaxSim(v, _keepCentroids);
        float unsureSim = MaxSim(v, _unsureCentroids);

        return (new[] { "churn", "keep", "unsure" }, new[] { churnProto, keepSim, unsureSim });
    }

    /// <summary>
    /// Gets a human-readable explanation for the current model decision, including gating and threshold checks.
    /// </summary>
    /// <param name="input">The user input text to explain.</param>
    /// <returns>A textual explanation of the model's scoring and decision bars.</returns>
    public string Explain(string input)
    {
        float[] v = _embedder.Embed(input ?? string.Empty);
        v = ApplyMeanCenter(v, _globalMean);

        float churnProto = MaxSim(v, _churnCentroids);
        float keepSim = MaxSim(v, _keepCentroids);
        float unsureSim = MaxSim(v, _unsureCentroids);
        float vote = ComputeChurnVote(v, K_NEIGHBORS);
        float churnScore = (ALPHA * churnProto) + ((1 - ALPHA) * vote);

        Span<float> sims = [churnProto, keepSim, unsureSim];
        (int topIdx, int secondIdx, float top, float second) = VectorMath.Top2(sims);
        float margin = top - second;

        string topLabel = topIdx switch
        {
            0 => "churn",
            1 => "keep",
            _ => "unsure",
        };

        bool cue = ChurnCueRegex().IsMatch(input ?? string.Empty);
        bool thrOK = (churnScore + EPS) >= _churnMinSim;
        bool marOK = (topIdx == 0) ? ((margin + EPS) >= _minMargin) : false;

        string reason;
        if (!cue)
        {
            reason = "blocked: cue=false";
        }
        else if (topIdx != 0)
        {
            reason = $"blocked: top='{topLabel}' (top={top:F2}) > churnProto({churnProto:F2})";
        }
        else if (!thrOK)
        {
            reason = $"blocked: blended threshold (score={churnScore:F2} < min={_churnMinSim:F2})";
        }
        else if (!marOK)
        {
            reason = $"blocked: margin (margin={margin:F2} < min={_minMargin:F2})";
        }
        else
        {
            reason = "churn: passed cue, threshold, and margin";
        }

        return
            $"Explain: cue={(cue ? "true" : "false")}, " +
            $"proto={churnProto:F4}, vote={vote:F2}, blended={churnScore:F4}, " +
            $"keep={keepSim:F4}, unsure={unsureSim:F4}, top={topLabel}, margin={margin:F4}, " +
            $"bars(thr={_churnMinSim:F4}, minMargin={_minMargin:F4}, eps={EPS:G}) -> {reason}";
    }

    /// <summary>
    /// Applies mean-centering using the provided global mean (if any) and re-normalizes the vector.
    /// </summary>
    /// <param name="v">The input vector to center.</param>
    /// <param name="mu">The global mean vector; if <c>null</c>, returns the original vector.</param>
    /// <returns>The centered and L2-normalized vector (or the original if no mean is provided).</returns>
    private static float[] ApplyMeanCenter(float[] v, float[]? mu)
    {
        if (mu is null)
        {
            return v;
        }

        float[] outv = (float[])v.Clone();
        double sum = 0;
        for (int i = 0; i < outv.Length; i++)
        {
            outv[i] -= mu[i];
            sum += (double)outv[i] * outv[i];
        }

        double norm = Math.Sqrt(sum);
        if (norm > 0)
        {
            float inv = (float)(1.0 / norm);
            for (int i = 0; i < outv.Length; i++)
            {
                outv[i] *= inv;
            }
        }

        return outv;
    }

    /// <summary>
    /// Gets the maximum dot-product similarity to any centroid within a label group.
    /// </summary>
    /// <param name="v">The (centered) query vector.</param>
    /// <param name="centroids">The centroids for a particular label.</param>
    /// <returns>The maximum similarity.</returns>
    private static float MaxSim(ReadOnlySpan<float> v, float[][] centroids)
    {
        float best = float.NegativeInfinity;
        for (int i = 0; i < centroids.Length; i++)
        {
            float s = VectorMath.Dot(v, centroids[i]);
            if (s > best)
            {
                best = s;
            }
        }

        return best;
    }

    /// <summary>
    /// Loads an optional seed bank from an embedded resource and returns the seed vectors and labels.
    /// </summary>
    /// <param name="asm">The assembly that contains the embedded seed bank resource.</param>
    /// <returns>An array of (vector, label) seed entries; an empty array if unavailable.</returns>
    private static (float[] V, string Label)[] LoadSeeds(Assembly asm)
    {
        try
        {
            string? name = asm
                .GetManifestResourceNames()
                .FirstOrDefault(n => n.EndsWith("." + SeedBankResourceName, StringComparison.OrdinalIgnoreCase));

            if (name is null)
            {
                return Array.Empty<(float[] V, string Label)>();
            }

            using Stream? s = asm.GetManifestResourceStream(name);
            if (s is null)
            {
                return Array.Empty<(float[] V, string Label)>();
            }

            using JsonDocument doc = JsonDocument.Parse(s);
            if (!doc.RootElement.TryGetProperty("seeds", out JsonElement arr))
            {
                return Array.Empty<(float[] V, string Label)>();
            }

            List<(float[] V, string Label)> list = new List<(float[] V, string Label)>(arr.GetArrayLength());
            foreach (JsonElement e in arr.EnumerateArray())
            {
                string label = e.GetProperty("label").GetString() ?? string.Empty;
                JsonElement vecEl = e.GetProperty("vector");
                float[] vec = new float[vecEl.GetArrayLength()];
                int i = 0;
                foreach (JsonElement el in vecEl.EnumerateArray())
                {
                    vec[i++] = el.GetSingle();
                }

                list.Add((vec, label));
            }

            return list.ToArray();
        }
        catch
        {
            return Array.Empty<(float[] V, string Label)>();
        }
    }

    /// <summary>
    /// Computes the fraction of churn labels among the top-k nearest seeds (k-NN vote).
    /// </summary>
    /// <param name="v">The (centered) query vector.</param>
    /// <param name="k">The number of nearest seeds to consider.</param>
    /// <returns>A value in [0,1] representing the fraction of churn in the top-k.</returns>
    private float ComputeChurnVote(ReadOnlySpan<float> v, int k)
    {
        if (_seeds.Length == 0 || k <= 0)
        {
            return 0f;
        }

        // Compute similarities to all seeds; pick top-k.
        List<(float Sim, string Label)> sims = new List<(float Sim, string Label)>(_seeds.Length);
        foreach ((float[] V, string Label) seed in _seeds)
        {
            sims.Add((VectorMath.Dot(v, seed.V), seed.Label));
        }

        sims.Sort((a, b) => b.Sim.CompareTo(a.Sim));
        int kk = Math.Min(k, sims.Count);

        int churnCount = 0;
        for (int i = 0; i < kk; i++)
        {
            if (string.Equals(sims[i].Label, "churn", StringComparison.OrdinalIgnoreCase))
            {
                churnCount++;
            }
        }

        return (float)churnCount / Math.Max(1, kk);
    }
}
