
using System.Text;

namespace Genova.ChurnDetector.Terminal;

internal class Program
{
    // Toggle automated mode here, or pass --auto on the command line.
    private static bool _automated = false;
    private static bool _verbose = false;

    private static void Main(string[] args)
    {
        if (args.Any(a => string.Equals(a, "--auto", StringComparison.OrdinalIgnoreCase)))
            _automated = true;

        var detector = new Detector();

        if (_automated)
        {
            RunAutomated(detector);
            return;
        }

        Console.WriteLine("Churn Signals — type a message (or 'exit').");
        Console.WriteLine();

        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine();
            if (input is null) break;
            if (input.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

            var (label, confidence) = detector.GetSignal(input);
            Console.WriteLine($"Label: {label} | Confidence: {confidence:F2}");

            if (_verbose)
            {
                // Show raw centroid similarities (churn/keep/unsure)
                var (labels, sims) = detector.GetRawSimilarities(input);
                Console.WriteLine($"Scores: {labels[0]}={sims[0]:F2}, {labels[1]}={sims[1]:F2}, {labels[2]}={sims[2]:F2}");
                Console.WriteLine(detector.Explain(input));
            }
        }

        Console.WriteLine("Bye.");
    }

    // -------------------- Automated evaluation --------------------

    private static void RunAutomated(Detector detector)
    {
        // Locate the CSV:
        // 1) ./Data/training.seed.csv next to the Terminal executable
        // 2) ../../../../Genova.ChurnDetector.Training/Data/training.seed.csv (relative to bin/)
        // 3) GENOVA_TRAIN_CSV env var (if provided)
        var csvPath = ResolveCsvPath();

        var rows = ReadCsv(csvPath).ToList(); // (text, label)
        if (rows.Count == 0)
        {
            Console.Error.WriteLine($"No rows found in CSV: {csvPath}");
            return;
        }

        // Tally per class
        var counts = new Dictionary<string, (int total, int correct)>(StringComparer.OrdinalIgnoreCase)
        {
            ["churn"] = (0, 0),
            ["keep"] = (0, 0),
            ["unsure"] = (0, 0)
        };

        foreach (var (text, label) in rows)
        {
            if (!counts.ContainsKey(label)) continue; // ignore unexpected labels

            var (pred, _) = detector.GetSignal(text);

            // Ground truth mapping:
            // - churn  -> expect "churn"
            // - keep   -> expect "not_churn"
            // - unsure -> expect "not_churn"
            bool isCorrect =
                (label.Equals("churn", StringComparison.OrdinalIgnoreCase) && pred == "churn") ||
                ((label.Equals("keep", StringComparison.OrdinalIgnoreCase) || label.Equals("unsure", StringComparison.OrdinalIgnoreCase)) && pred == "not_churn");

            var (total, correct) = counts[label];
            counts[label] = (total + 1, correct + (isCorrect ? 1 : 0));
        }

        // Print exactly three lines (counts + percentage correct)
        PrintClassResult("churn", counts);
        PrintClassResult("keep", counts);
        PrintClassResult("unsure", counts);
    }

    private static void PrintClassResult(string cls, Dictionary<string, (int total, int correct)> counts)
    {
        var (total, correct) = counts.TryGetValue(cls, out var v) ? v : (0, 0);
        var pct = total > 0 ? (100.0 * correct / total) : 0.0;
        Console.WriteLine($"{cls}: {total} entries — {pct:F1}% correct");
    }

    private static string ResolveCsvPath()
    {
        var exe = AppContext.BaseDirectory;

        var candidates = new[]
        {
            Path.Combine(exe, "Data", "training.seed.csv"),
            // From .../Genova.ChurnDetector.Terminal/bin/Debug/net8.0/
            Path.GetFullPath(Path.Combine(exe, "../../../../ChurnDetector.Training/Input/training.seed.csv")),
            Environment.GetEnvironmentVariable("GENOVA_TRAIN_CSV")
        }.Where(p => !string.IsNullOrWhiteSpace(p)).ToList();

        foreach (var p in candidates)
        {
            if (File.Exists(p)) return p;
        }

        throw new FileNotFoundException("Could not find training.seed.csv. Checked:\n" + string.Join("\n", candidates));
    }

    // -------------------- Minimal CSV reader (RFC4180-ish) --------------------

    private static IEnumerable<(string Text, string Label)> ReadCsv(string path)
    {
        using var sr = new StreamReader(path, DetectEncoding(path), detectEncodingFromByteOrderMarks: true);

        // Header
        var header = sr.ReadLine();
        if (header is null) yield break;

        var head = ParseCsvLine(header);
        if (head.Count < 2 || !head[0].Equals("text", StringComparison.OrdinalIgnoreCase) || !head[1].Equals("label", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("CSV header must be exactly: text,label");

        // Rows
        string? line;
        while ((line = sr.ReadLine()) is not null)
        {
            if (line.Length == 0) continue;
            var fields = ParseCsvLine(line);
            if (fields.Count < 2) continue;

            var text = fields[0] ?? string.Empty;
            var label = (fields[1] ?? string.Empty).Trim().ToLowerInvariant();
            yield return (text, label);
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
                    // Escaped quote?
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i++; // skip the second quote
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
                    result.Add(sb.ToString());
                    sb.Clear();
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
}

