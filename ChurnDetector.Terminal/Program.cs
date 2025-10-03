// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace Genova.ChurnDetector.Terminal;

/// <summary>
/// Hosts the console application used to interact with the churn detector.
/// Supports interactive mode and an automated evaluation mode.
/// </summary>
[SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "Conflicting naming rules.")]
internal class Program
{
    // Toggle automated mode here, or pass --auto on the command line.
    private static bool _automated = true;

    // Toggle verbose, developer-oriented output (raw scores, explanations).
    private static readonly bool _verbose = false;

    /// <summary>
    /// Entry point for the console application. Parses command-line arguments and starts the
    /// churn detector in either interactive mode or automated evaluation mode.
    /// </summary>
    /// <param name="args">
    /// The command-line arguments. Specify <c>--auto</c> to run automated evaluation over the
    /// training CSV and print per-class accuracy; omit to start an interactive REPL that reads
    /// user input and prints detection results.
    /// </param>
    /// <remarks>
    /// In interactive mode, the application prompts for a line of text and prints the predicted
    /// label and confidence for each input. In automated mode, it evaluates all rows in the
    /// training dataset and prints summary accuracy for the <c>churn</c>, <c>keep</c>, and
    /// <c>unsure</c> classes.
    /// </remarks>
    private static void Main(string[] args)
    {
        if (args.Any(a => string.Equals(a, "--auto", StringComparison.OrdinalIgnoreCase)))
        {
            _automated = true;
        }

        Detector detector = new Detector();

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
            string? input = Console.ReadLine();
            if (input is null)
            {
                break;
            }

            if (input.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            (string label, float confidence) = detector.GetSignal(input);
            Console.WriteLine($"Label: {label} | Confidence: {confidence:F2}");

            if (_verbose)
            {
                // Show raw centroid similarities (churn/keep/unsure)
                (string[] labels, float[] sims) = detector.GetRawSimilarities(input);
                Console.WriteLine($"Scores: {labels[0]}={sims[0]:F2}, {labels[1]}={sims[1]:F2}, {labels[2]}={sims[2]:F2}");
                Console.WriteLine(detector.Explain(input));
            }
        }

        Console.WriteLine("Bye.");
    }

    /// <summary>
    /// Runs the detector against all rows in the training CSV and prints per-class accuracy.
    /// </summary>
    /// <param name="detector">The detector instance to evaluate.</param>
    private static void RunAutomated(Detector detector)
    {
        // Locate the CSV:
        // 1) ./Data/training.seed.csv next to the Terminal executable
        // 2) ../../../../Genova.ChurnDetector.Training/Input/training.seed.csv (relative to bin/)
        // 3) GENOVA_TRAIN_CSV env var (if provided)
        string csvPath = ResolveCsvPath();

        List<(string Text, string Label)> rows = ReadCsv(csvPath).ToList();
        if (rows.Count == 0)
        {
            Console.Error.WriteLine($"No rows found in CSV: {csvPath}");
            return;
        }

        // Tally per class
        Dictionary<string, (int total, int correct)> counts =
            new Dictionary<string, (int total, int correct)>(StringComparer.OrdinalIgnoreCase)
            {
                ["churn"] = (0, 0),
                ["keep"] = (0, 0),
                ["unsure"] = (0, 0)
            };

        foreach ((string Text, string Label) row in rows)
        {
            string text = row.Text;
            string label = row.Label;

            if (!counts.ContainsKey(label))
            {
                continue; // ignore unexpected labels
            }

            (string pred, float _) = detector.GetSignal(text);

            // Ground truth mapping:
            // - churn  -> expect "churn"
            // - keep   -> expect "not_churn"
            // - unsure -> expect "not_churn"
            bool isCorrect =
                (label.Equals("churn", StringComparison.OrdinalIgnoreCase) && pred == "churn") ||
                ((label.Equals("keep", StringComparison.OrdinalIgnoreCase) || label.Equals("unsure", StringComparison.OrdinalIgnoreCase)) && pred == "not_churn");

            (int total, int correct) = counts[label];
            counts[label] = (total + 1, correct + (isCorrect ? 1 : 0));
        }

        // Print exactly three lines (counts + percentage correct)
        PrintClassResult("churn", counts);
        PrintClassResult("keep", counts);
        PrintClassResult("unsure", counts);
    }

    /// <summary>
    /// Prints a single summary line for a given class showing count and percent correct.
    /// </summary>
    /// <param name="cls">The class key ("churn", "keep", or "unsure").</param>
    /// <param name="counts">The class counters map.</param>
    private static void PrintClassResult(string cls, Dictionary<string, (int total, int correct)> counts)
    {
        (int total, int correct) = counts.TryGetValue(cls, out (int total, int correct) v) ? v : (0, 0);
        double pct = total > 0 ? (100.0 * correct / total) : 0.0;
        Console.WriteLine($"{cls}: {total} entries — {pct:F1}% correct");
    }

    /// <summary>
    /// Resolves the path to the training CSV by checking several common locations and an environment variable.
    /// </summary>
    /// <returns>The absolute path to the CSV file.</returns>
    /// <exception cref="FileNotFoundException">Thrown when the CSV cannot be found in any location.</exception>
    private static string ResolveCsvPath()
    {
        string exe = AppContext.BaseDirectory;

        List<string> candidates = new List<string>
        {
            Path.Combine(exe, "Data", "training.seed.csv"),
            // From .../Genova.ChurnDetector.Terminal/bin/Debug/net8.0/
            Path.GetFullPath(Path.Combine(exe, "../../../../ChurnDetector.Training/Input/training.seed.csv"))
        };

        string? env = Environment.GetEnvironmentVariable("GENOVA_TRAIN_CSV");
        if (!string.IsNullOrWhiteSpace(env))
        {
            candidates.Add(env);
        }

        foreach (string p in candidates)
        {
            if (File.Exists(p))
            {
                return p;
            }
        }

        throw new FileNotFoundException("Could not find training.seed.csv. Checked:\n" + string.Join("\n", candidates));
    }

    /// <summary>
    /// Reads (text, label) rows from a CSV file that has a header of exactly "text,label".
    /// </summary>
    /// <param name="path">The path to the CSV file.</param>
    /// <returns>An enumerable of (Text, Label) tuples.</returns>
    private static IEnumerable<(string Text, string Label)> ReadCsv(string path)
    {
        using (StreamReader sr = new StreamReader(path, DetectEncoding(path), true))
        {
            // Header
            string? header = sr.ReadLine();
            if (header is null)
            {
                yield break;
            }

            List<string> head = ParseCsvLine(header);
            if (head.Count < 2 || !head[0].Equals("text", StringComparison.OrdinalIgnoreCase) || !head[1].Equals("label", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("CSV header must be exactly: text,label");
            }

            // Rows
            string? line;
            while ((line = sr.ReadLine()) is not null)
            {
                if (line.Length == 0)
                {
                    continue;
                }

                List<string> fields = ParseCsvLine(line);
                if (fields.Count < 2)
                {
                    continue;
                }

                string text = fields[0] ?? string.Empty;
                string label = (fields[1] ?? string.Empty).Trim().ToLowerInvariant();
                yield return (text, label);
            }
        }
    }

    /// <summary>
    /// Detects the encoding of a text file using the BOM if present, else UTF-8 without BOM.
    /// </summary>
    /// <param name="path">The file path.</param>
    /// <returns>An <see cref="Encoding"/> suitable for reading the file.</returns>
    private static UTF8Encoding DetectEncoding(string path)
    {
        using (FileStream fs = File.OpenRead(path))
        {
            if (fs.Length >= 3)
            {
                Span<byte> bom = stackalloc byte[3];
                _ = fs.Read(bom);
                if (bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF)
                {
                    return new UTF8Encoding(encoderShouldEmitUTF8Identifier: true);
                }
            }
        }

        return new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
    }

    /// <summary>
    /// Parses a single CSV line into fields (RFC 4180-ish), handling quotes and commas.
    /// </summary>
    /// <param name="line">The CSV line.</param>
    /// <returns>A list of parsed fields.</returns>
    private static List<string> ParseCsvLine(string line)
    {
        List<string> result = [];
        if (line is null)
        {
            return result;
        }

        StringBuilder sb = new StringBuilder();
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
