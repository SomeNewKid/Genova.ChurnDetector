// File: Genova.ChurnDetector.Training/Program.cs
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Genova.ChurnDetector.Training;

internal static class Program
{
    private const string DefaultDatasetFileName = "training.seed.csv"; // header: text,label
    private static readonly string[] AllowedLabels = new[] { "churn", "keep", "unsure" };

    // ---- Hyperparameter grid ----
    private static readonly int[] CandidateDims = new[] { 256, 384, 512 };
    private static readonly float[] CandidateChurnMinSims = new[] { 0.38f, 0.40f, 0.42f, 0.44f, 0.46f, 0.48f, 0.50f, 0.52f };
    private static readonly float[] CandidateMargins = new[] { 0.00f, 0.01f, 0.02f };

    // k-means for churn
    private const int KChurn = 4;

    // Seed bank caps (centered vectors)
    private const int SeedCapChurn = 200;
    private const int SeedCapUnsure = 200;
    private const int SeedCapKeep = 120;

    // k-NN blend (MUST MATCH Detector.cs to align selection with runtime)
    private const int K_NEIGHBORS = 9;
    private const float ALPHA = 0.60f; // blended = ALPHA*proto + (1-ALPHA)*vote

    private const int SeedSplit = 42;

    private sealed record Row(string Text, string Label);
    private sealed record VecRow(float[] V, string Label, string Text);

    private sealed record Metrics(
        int TP, int FP, int FN, int TN,
        float Precision, float Recall, float F1, int Total);

    private sealed record TrialResult(
        int Dim, float ChurnMinSim, float MinMargin,
        Metrics DevMetrics);

    private sealed record Report(
        TrialResult Champion,
        Metrics TestMetrics,
        int TrainCount, int DevCount, int TestCount,
        string[] Labels, DateTime CreatedUtc);

    static int Main(string[] args)
    {
        try
        {
            var root = AppContext.BaseDirectory;

            // Input dataset (your runs use .../bin/.../Input/training.seed.csv)
            var datasetPath = Path.Combine(root, "Input", DefaultDatasetFileName);
            if (!File.Exists(datasetPath))
                throw new FileNotFoundException($"CSV dataset not found at: {datasetPath}");

            // Output directory for artifacts (defaults to the library's Data folder)
            string outDir = (args.Length >= 1 && !string.IsNullOrWhiteSpace(args[0]))
                ? args[0]
                : Path.GetFullPath(Path.Combine(root, "../../../../ChurnDetector/Data"));
            Directory.CreateDirectory(outDir);

            Console.WriteLine($"Dataset (CSV): {datasetPath}");
            Console.WriteLine($"Artifacts out: {outDir}");
            Console.WriteLine("Reading dataset...");

            var rawRows = ReadCsv(datasetPath).ToList();
            if (rawRows.Count == 0)
            {
                Console.Error.WriteLine("No rows found.");
                return 2;
            }

            // Normalize and filter to allowed labels
            var rows = rawRows
                .Select(r => new Row(r.Text, (r.Label ?? "").Trim().ToLowerInvariant()))
                .Where(r => AllowedLabels.Contains(r.Label))
                .ToList();

            if (rows.Count == 0)
            {
                Console.Error.WriteLine("No rows with allowed labels (churn|keep|unsure).");
                return 2;
            }

            // Stratified split: 70/15/15 (train/dev/test)
            var (trainRows, devRows, testRows) = StratifiedSplit(rows, 0.70, 0.15, 0.15, SeedSplit);
            Console.WriteLine($"Split sizes: train={trainRows.Count}, dev={devRows.Count}, test={testRows.Count}");

            // Iterate over hyperparameters; evaluate on dev; pick champion
            var trials = new List<TrialResult>();
            foreach (var dim in CandidateDims)
            {
                Console.WriteLine($"Embedding dim={dim} ...");

                var embedder = new LocalHashEmbedder(dim);

                // Embed once per split
                var trainVecs = EmbedRows(trainRows, embedder);
                var devVecs = EmbedRows(devRows, embedder);
                var testVecs = EmbedRows(testRows, embedder); // reserved for final eval

                // ---- MEAN-CENTER for DEV evaluation ----
                var muTrain = ComputeGlobalMean(trainVecs, dim);
                var trainC = CenterRows(trainVecs, muTrain);
                var devC = CenterRows(devVecs, muTrain);

                // ---- Seed bank for DEV blending (built from TRAIN, centered) ----
                var seedBankDev = BuildSeedBank(trainC, SeedCapChurn, SeedCapUnsure, SeedCapKeep);

                // ---- Build prototypes on centered TRAIN ----
                var churnTrainRows = trainC.Where(v => v.Label == "churn").ToList();
                var keepTrain = trainC.Where(v => v.Label == "keep").Select(v => v.V).ToList();
                var unsureTrain = trainC.Where(v => v.Label == "unsure").Select(v => v.V).ToList();

                var churnProtosKMeans = KMeansCosine(
                    churnTrainRows.Select(v => v.V).ToList(),
                    Math.Min(KChurn, Math.Max(1, churnTrainRows.Count)), dim);

                var churnProtosAnchors = BuildPatternPrototypesFromVecs(churnTrainRows, dim, minSupport: 6);

                var churnProtosTrain = churnProtosKMeans.Concat(churnProtosAnchors).ToArray();
                var keepCentroidTrain = ComputeCentroid(keepTrain, dim);
                var unsrCentroidTrain = ComputeCentroid(unsureTrain, dim);

                foreach (var thr in CandidateChurnMinSims)
                {
                    foreach (var margin in CandidateMargins)
                    {
                        var devMetrics = EvaluateBinaryChurnBlended(
                            devC, churnProtosTrain, keepCentroidTrain, unsrCentroidTrain,
                            seedBankDev, thr, margin,
                            Detector.ChurnCueRegex, Detector.NegationNearCueRegex);
                        trials.Add(new TrialResult(dim, thr, margin, devMetrics));
                    }
                }
            }

            // ---- Choose champion: highest recall subject to precision & specificity floors ----
            static float Specificity(Metrics m)
            {
                var denom = (m.TN + m.FP);
                return denom > 0 ? (float)m.TN / denom : 1f;
            }

            var pool = trials
                .Where(t => t.DevMetrics.Precision >= 0.95f && Specificity(t.DevMetrics) >= 0.95f)
                .ToList();

            if (pool.Count == 0) pool = trials.Where(t => t.DevMetrics.Precision >= 0.95f).ToList();

            TrialResult champion = (pool.Count == 0)
                ? trials.OrderByDescending(t => t.DevMetrics.F1).ThenByDescending(t => t.DevMetrics.Precision).First()
                : pool.OrderByDescending(t => t.DevMetrics.Recall).ThenByDescending(t => t.DevMetrics.F1).First();

            Console.WriteLine();
            Console.WriteLine("Champion (DEV):");
            Console.WriteLine($"  dim={champion.Dim}  churn_min_sim={champion.ChurnMinSim:F2}  min_margin={champion.MinMargin:F2}");
            Console.WriteLine($"  DEV: P={champion.DevMetrics.Precision:F3}  R={champion.DevMetrics.Recall:F3}  F1={champion.DevMetrics.F1:F3}");

            // ---- Refit on TRAIN+DEV with champion dim (centered) ----
            var embedderFinal = new LocalHashEmbedder(champion.Dim);
            var trainDevRows = trainRows.Concat(devRows).ToList();
            var trainDevVecs = EmbedRows(trainDevRows, embedderFinal);

            var muFinal = ComputeGlobalMean(trainDevVecs, champion.Dim);
            var trainDevC = CenterRows(trainDevVecs, muFinal);

            var churnTrainDevRows = trainDevC.Where(v => v.Label == "churn").ToList();
            var keepTrainDevVecs = trainDevC.Where(v => v.Label == "keep").Select(v => v.V).ToList();
            var unsureTrainDevVecs = trainDevC.Where(v => v.Label == "unsure").Select(v => v.V).ToList();

            var churnProtosFinalKM = KMeansCosine(
                churnTrainDevRows.Select(v => v.V).ToList(),
                Math.Min(KChurn, Math.Max(1, churnTrainDevRows.Count)), champion.Dim);

            var churnProtosFinalAnchors = BuildPatternPrototypesFromVecs(churnTrainDevRows, champion.Dim, minSupport: 6);
            var churnProtosFinal = churnProtosFinalKM.Concat(churnProtosFinalAnchors).ToArray();

            var keepCentroidFinal = ComputeCentroid(keepTrainDevVecs, champion.Dim);
            var unsrCentroidFinal = ComputeCentroid(unsureTrainDevVecs, champion.Dim);

            // Final seed bank (centered TRAIN+DEV) for runtime
            var seedBankFinal = BuildSeedBank(trainDevC, SeedCapChurn, SeedCapUnsure, SeedCapKeep);

            // ---- Evaluate ONCE on TEST (centered with μ_final) ----
            var testVecs2 = EmbedRows(testRows, embedderFinal);
            var testC = CenterRows(testVecs2, muFinal);

            var testMetrics = EvaluateBinaryChurnBlended(
                testC, churnProtosFinal, keepCentroidFinal, unsrCentroidFinal,
                seedBankFinal, champion.ChurnMinSim, champion.MinMargin,
                Detector.ChurnCueRegex, Detector.NegationNearCueRegex);

            Console.WriteLine("Test metrics (held-out):");
            Console.WriteLine($"  TEST: P={testMetrics.Precision:F3}  R={testMetrics.Recall:F3}  F1={testMetrics.F1:F3}");
            Console.WriteLine();
            Console.WriteLine($"[Anchors] churn prototypes: kmeans={churnProtosFinalKM.Length}, anchors={churnProtosFinalAnchors.Length}, total={churnProtosFinal.Length}");

            // ---- Persist artifacts (manifest + centroids + seeds) ----
            var labelsArr = AllowedLabels;

            // Include global mean as a special centroid "__global_mean" (not normalized)
            var centroidsFinal = new List<dynamic>();
            foreach (var v in churnProtosFinal) centroidsFinal.Add(new { label = "churn", vector = v });
            centroidsFinal.Add(new { label = "keep", vector = keepCentroidFinal });
            centroidsFinal.Add(new { label = "unsure", vector = unsrCentroidFinal });
            centroidsFinal.Add(new { label = "__global_mean", vector = muFinal });

            WriteArtifacts(outDir, champion, centroidsFinal.ToArray(), labelsArr);

            // Seed bank file (centered)
            WriteSeedBank(outDir, seedBankFinal);

            // ---- Save reports ----
            var report = new Report(
                Champion: champion,
                TestMetrics: testMetrics,
                TrainCount: trainRows.Count,
                DevCount: devRows.Count,
                TestCount: testRows.Count,
                Labels: labelsArr,
                CreatedUtc: DateTime.UtcNow
            );
            File.WriteAllText(Path.Combine(outDir, "training.report.json"),
                JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));

            using (var log = File.CreateText(Path.Combine(outDir, "experiments.jsonl")))
            {
                foreach (var t in trials)
                {
                    var line = JsonSerializer.Serialize(t);
                    log.WriteLine(line);
                }
            }

            Console.WriteLine("Wrote:");
            Console.WriteLine(" - " + Path.Combine(outDir, Detector.ManifestResourceName));
            Console.WriteLine(" - " + Path.Combine(outDir, Detector.CentroidsResourceName));
            Console.WriteLine(" - " + Path.Combine(outDir, Detector.SeedBankResourceName));
            Console.WriteLine(" - " + Path.Combine(outDir, "training.report.json"));
            Console.WriteLine(" - " + Path.Combine(outDir, "experiments.jsonl"));

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.ToString());
            return 1;
        }
    }

    // --------- Blended evaluator (matches runtime) ---------
    private static Metrics EvaluateBinaryChurnBlended(
        List<VecRow> evalSet,
        float[][] churnProtos,
        float[] keepC,
        float[] unsureC,
        (List<float[]> churn, List<float[]> unsure, List<float[]> keep) seeds,
        float churnMinSim,
        float minMargin,
        Regex cueRegex,
        Regex negRegex)
    {
        int tp = 0, fp = 0, fn = 0, tn = 0;

        foreach (var ex in evalSet)
        {
            // Prototype similarities (ranking)
            float churnProto = float.NegativeInfinity;
            for (int i = 0; i < churnProtos.Length; i++)
                churnProto = Math.Max(churnProto, VectorMath.Dot(ex.V, churnProtos[i]));

            float keepSim = VectorMath.Dot(ex.V, keepC);
            float unsureSim = VectorMath.Dot(ex.V, unsureC);

            Span<float> sims = stackalloc float[3] { churnProto, keepSim, unsureSim };
            var (topIdx, secondIdx, top, second) = VectorMath.Top2(sims);
            var margin = top - second;

            // k-NN vote (centered space)
            float vote = ComputeChurnVote(ex.V, seeds, K_NEIGHBORS);
            float blended = ALPHA * churnProto + (1 - ALPHA) * vote;

            // Gate like runtime
            bool hasCue = cueRegex.IsMatch(ex.Text ?? string.Empty);
            bool negatesCue = negRegex.IsMatch(ex.Text ?? string.Empty);

            bool predictChurn =
                hasCue && !negatesCue &&
                (topIdx == 0) &&
                (blended >= churnMinSim) &&
                (margin >= minMargin);

            bool actualChurn = ex.Label == "churn";

            if (predictChurn && actualChurn) tp++;
            else if (predictChurn && !actualChurn) fp++;
            else if (!predictChurn && actualChurn) fn++;
            else tn++;
        }

        var precision = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0f;
        var recall = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0f;
        var f1 = (precision + recall) > 0 ? 2f * precision * recall / (precision + recall) : 0f;
        return new Metrics(tp, fp, fn, tn, precision, recall, f1, evalSet.Count);
    }

    private static float ComputeChurnVote(
        ReadOnlySpan<float> v,
        (List<float[]> churn, List<float[]> unsure, List<float[]> keep) seeds,
        int k)
    {
        // Collect sims
        var sims = new List<(float sim, string label)>(seeds.churn.Count + seeds.unsure.Count + seeds.keep.Count);
        foreach (var s in seeds.churn) sims.Add((VectorMath.Dot(v, s), "churn"));
        foreach (var s in seeds.unsure) sims.Add((VectorMath.Dot(v, s), "unsure"));
        foreach (var s in seeds.keep) sims.Add((VectorMath.Dot(v, s), "keep"));

        // Top-k
        sims.Sort((a, b) => b.sim.CompareTo(a.sim));
        int kk = Math.Min(k, sims.Count);
        int churnCount = 0;
        for (int i = 0; i < kk; i++)
            if (sims[i].label == "churn") churnCount++;

        return (float)churnCount / Math.Max(1, kk);
    }

    // --------- Pattern anchors (centered space) ---------
    private static float[][] BuildPatternPrototypesFromVecs(
        IEnumerable<VecRow> churnVecs, int dim, int minSupport = 6)
    {
        var patterns = new (string name, Regex rx)[]
        {
            ("close_account",
                new Regex(@"\bclose\s+(?:my|our|the)?\s*account\b",
                    RegexOptions.IgnoreCase | RegexOptions.Compiled)),
            ("stop_subscription",
                new Regex(@"\bstop\s+(?:my|our|the)?\s*(?:subscription|service|plan|membership|auto\s*renew(?:al)?)\b",
                    RegexOptions.IgnoreCase | RegexOptions.Compiled)),
            ("end_subscription",
                new Regex(@"\bend\s+(?:my|our|the)?\s*(?:account|subscription|service|plan|membership)\b",
                    RegexOptions.IgnoreCase | RegexOptions.Compiled)),
            ("cancel_or_terminate",
                new Regex(@"\b(cancel|terminate|unsubscribe)\b.*\b(account|subscription|service|plan|membership)\b|\b(account|subscription|service|plan|membership)\b.*\b(cancel|terminate|unsubscribe)\b",
                    RegexOptions.IgnoreCase | RegexOptions.Compiled)),
        };

        var anchors = new List<float[]>();

        foreach (var (_, rx) in patterns)
        {
            var bag = new List<float[]>();
            foreach (var row in churnVecs)
            {
                var txt = row.Text ?? string.Empty;
                if (rx.IsMatch(txt)) bag.Add(row.V);
            }

            if (bag.Count >= minSupport)
                anchors.Add(ComputeCentroid(bag, dim)); // mean + normalize (already centered inputs)
        }

        return anchors.ToArray();
    }

    // --------- Mean-centering helpers ---------
    private static float[] ComputeGlobalMean(List<VecRow> rows, int dim)
    {
        var mu = new float[dim];
        if (rows.Count == 0) return mu;
        foreach (var r in rows)
            for (int i = 0; i < dim; i++) mu[i] += r.V[i];
        for (int i = 0; i < dim; i++) mu[i] /= rows.Count;
        return mu;
    }

    private static List<VecRow> CenterRows(List<VecRow> rows, float[] mu)
    {
        var list = new List<VecRow>(rows.Count);
        foreach (var r in rows)
        {
            var v = (float[])r.V.Clone();
            CenterAndNormalizeInPlace(v, mu);
            list.Add(new VecRow(v, r.Label, r.Text));
        }
        return list;
    }

    private static void CenterAndNormalizeInPlace(float[] v, float[] mu)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++)
        {
            v[i] -= mu[i];
            sum += (double)v[i] * v[i];
        }
        var norm = Math.Sqrt(sum);
        if (norm > 0)
        {
            var inv = (float)(1.0 / norm);
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
        }
    }

    // --------- k-means (cosine) & centroid ---------
    private static float[][] KMeansCosine(IReadOnlyList<float[]> vecs, int k, int dim, int maxIters = 50, int seed = 123)
    {
        if (vecs.Count == 0 || k <= 1) return new[] { ComputeCentroid(vecs, dim) };

        var rnd = new Random(seed);

        // Init: pick k random points
        var init = vecs.OrderBy(_ => rnd.Next()).Take(k).Select(v => (float[])v.Clone()).ToArray();
        for (int i = 0; i < init.Length; i++) VectorMath.L2Normalize(init[i]);

        var centroids = init;
        var assigns = new int[vecs.Count];

        for (int it = 0; it < maxIters; it++)
        {
            bool changed = false;

            // Assign
            for (int i = 0; i < vecs.Count; i++)
            {
                int best = 0; float bestSim = float.NegativeInfinity;
                for (int c = 0; c < centroids.Length; c++)
                {
                    var s = VectorMath.Dot(vecs[i], centroids[c]); // cosine (L2 normed)
                    if (s > bestSim) { bestSim = s; best = c; }
                }
                if (assigns[i] != best) { assigns[i] = best; changed = true; }
            }

            // Recompute
            var buckets = Enumerable.Range(0, k).Select(_ => new List<float[]>()).ToArray();
            for (int i = 0; i < vecs.Count; i++) buckets[assigns[i]].Add(vecs[i]);

            var newC = new float[k][];
            for (int c = 0; c < k; c++)
            {
                if (buckets[c].Count == 0)
                    newC[c] = (float[])vecs[rnd.Next(vecs.Count)].Clone();
                else
                    newC[c] = ComputeCentroid(buckets[c], dim);

                VectorMath.L2Normalize(newC[c]);
            }

            centroids = newC;
            if (!changed) break;
        }

        return centroids;
    }

    private static float[] ComputeCentroid(IEnumerable<float[]> vectors, int dim)
    {
        var sum = new float[dim];
        int n = 0;
        foreach (var v in vectors)
        {
            for (int i = 0; i < dim; i++) sum[i] += v[i];
            n++;
        }
        if (n == 0) return sum; // all zeros
        for (int i = 0; i < dim; i++) sum[i] /= n;
        VectorMath.L2Normalize(sum);
        return sum;
    }

    // --------- Seed bank I/O ---------
    private static void WriteSeedBank(string outDir, (List<float[]> churn, List<float[]> unsure, List<float[]> keep) seedBank)
    {
        var seeds = new List<object>(seedBank.churn.Count + seedBank.unsure.Count + seedBank.keep.Count);
        seeds.AddRange(seedBank.churn.Select(v => new { label = "churn", vector = v }));
        seeds.AddRange(seedBank.unsure.Select(v => new { label = "unsure", vector = v }));
        seeds.AddRange(seedBank.keep.Select(v => new { label = "keep", vector = v }));

        var payload = new { seeds };
        var path = Path.Combine(outDir, Detector.SeedBankResourceName);
        File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = false }));
    }

    private static (List<float[]> churn, List<float[]> unsure, List<float[]> keep)
        BuildSeedBank(List<VecRow> rowsCentered, int capChurn, int capUnsure, int capKeep)
    {
        var rnd = new Random(777);

        static List<float[]> Pick(IList<float[]> src, int cap, Random rnd)
        {
            var idx = Enumerable.Range(0, src.Count).OrderBy(_ => rnd.Next()).Take(Math.Min(cap, src.Count)).ToArray();
            var list = new List<float[]>(idx.Length);
            foreach (var i in idx) list.Add(src[i]);
            return list;
        }

        var churn = rowsCentered.Where(r => r.Label == "churn").Select(r => r.V).ToList();
        var unsure = rowsCentered.Where(r => r.Label == "unsure").Select(r => r.V).ToList();
        var keep = rowsCentered.Where(r => r.Label == "keep").Select(r => r.V).ToList();

        return (Pick(churn, capChurn, rnd), Pick(unsure, capUnsure, rnd), Pick(keep, capKeep, rnd));
    }

    // --------- CSV / split helpers ---------
    private static IEnumerable<Row> ReadCsv(string path)
    {
        using var sr = new StreamReader(path, DetectEncoding(path), detectEncodingFromByteOrderMarks: true);

        var header = sr.ReadLine();
        if (header is null) yield break;

        var head = ParseCsvLine(header);
        if (head.Count < 2 || !head[0].Equals("text", StringComparison.OrdinalIgnoreCase) || !head[1].Equals("label", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("CSV header must be exactly: text,label");

        string? line;
        while ((line = sr.ReadLine()) is not null)
        {
            if (line.Length == 0) continue;
            var fields = ParseCsvLine(line);
            if (fields.Count < 2) continue;

            var text = fields[0] ?? string.Empty;
            var label = fields[1] ?? string.Empty;
            yield return new Row(text, label);
        }
    }

    private static Encoding DetectEncoding(string path)
    {
        using var fs = File.OpenRead(path);
        if (fs.Length >= 3)
        {
            Span<byte> bom = stackalloc byte[3];
            _ = fs.Read(bom);
            if (bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF)
                return new UTF8Encoding(encoderShouldEmitUTF8Identifier: true);
        }
        return new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
    }

    private static List<string> ParseCsvLine(string line)
    {
        var result = new List<string>();
        if (line is null) return result;

        var sb = new StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];

            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"'); i++; // escaped quote
                    }
                    else
                    {
                        inQuotes = false; // closing quote
                    }
                }
                else
                {
                    sb.Append(c);
                }
            }
            else
            {
                if (c == ',')
                {
                    result.Add(sb.ToString()); sb.Clear();
                }
                else if (c == '"')
                {
                    inQuotes = true;
                }
                else
                {
                    sb.Append(c);
                }
            }
        }

        result.Add(sb.ToString());
        return result;
    }

    private static (List<Row> train, List<Row> dev, List<Row> test) StratifiedSplit(
        List<Row> rows, double pTrain, double pDev, double pTest, int seed)
    {
        var rng = new Random(seed);
        var byLabel = rows.GroupBy(r => r.Label).ToDictionary(g => g.Key, g => g.ToList());

        var train = new List<Row>();
        var dev = new List<Row>();
        var test = new List<Row>();

        foreach (var kv in byLabel)
        {
            var list = kv.Value;
            ShuffleInPlace(list, rng);

            int n = list.Count;
            int nTrain = (int)Math.Round(n * pTrain);
            int nDev = (int)Math.Round(n * pDev);
            int nTest = n - nTrain - nDev;

            if (nTrain < 1 && n > 0) nTrain = 1;
            if (nDev < 1 && n - nTrain > 1) nDev = 1;
            nTest = n - nTrain - nDev;

            train.AddRange(list.Take(nTrain));
            dev.AddRange(list.Skip(nTrain).Take(nDev));
            test.AddRange(list.Skip(nTrain + nDev).Take(nTest));
        }

        return (train, dev, test);
    }

    private static void ShuffleInPlace<T>(IList<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static void WriteArtifacts(
        string outDir,
        TrialResult champion,
        dynamic[] centroidsFinal,
        string[] labels)
    {
        Console.WriteLine($"champion.ChurnMinSim: {champion.ChurnMinSim}");
        Console.WriteLine($"champion.MinMargin: {champion.MinMargin}");

        var manifest = new
        {
            version = 1,
            created_utc = DateTime.UtcNow.ToString("o"),
            vector_dim = champion.Dim,
            labels = labels,
            thresholds = new { churn_min_similarity = champion.ChurnMinSim, min_margin = champion.MinMargin },
            embedder = new
            {
                type = "CharTrigramHashing",
                ngram = 3,
                dim = champion.Dim,
                hash = "FNV1A64",
                signed = true,
                lowercase = true,
                keep_alnum_space_only = true,
                padding_spaces = 2
            },
            blend = new { use_knn = true, k = K_NEIGHBORS, alpha = ALPHA }
        };

        var centroidsObj = new
        {
            centroids = centroidsFinal.Select(c => new { label = (string)c.label, vector = (float[])c.vector }).ToArray()
        };

        File.WriteAllText(Path.Combine(outDir, Detector.ManifestResourceName),
            JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));

        File.WriteAllText(Path.Combine(outDir, Detector.CentroidsResourceName),
            JsonSerializer.Serialize(centroidsObj, new JsonSerializerOptions { WriteIndented = false }));
    }

    private static List<VecRow> EmbedRows(List<Row> rows, LocalHashEmbedder embedder)
    {
        var outList = new List<VecRow>(rows.Count);
        foreach (var r in rows)
        {
            var v = embedder.Embed(r.Text ?? string.Empty);
            outList.Add(new VecRow(v, r.Label, r.Text));
        }
        return outList;
    }
}
