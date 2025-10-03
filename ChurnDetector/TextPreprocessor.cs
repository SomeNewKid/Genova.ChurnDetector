
using System.Text.RegularExpressions;

namespace Genova.ChurnDetector;

internal static partial class TextPreprocessor
{
    public static string Clean(string text)
    {
        if (text is null) return string.Empty;
        var t = text.ToLowerInvariant();
        t = NonAlnum().Replace(t, " ");
        t = MultiSpace().Replace(t, " ").Trim();
        return t;
    }

    [GeneratedRegex(@"[^a-z0-9 ]+", RegexOptions.Compiled)]
    private static partial Regex NonAlnum();
    [GeneratedRegex(@"\s+", RegexOptions.Compiled)]
    private static partial Regex MultiSpace();
}
