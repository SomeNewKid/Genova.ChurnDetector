// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Text.RegularExpressions;

namespace Genova.ChurnDetector;

/// <summary>
/// Provides basic normalization utilities for input text prior to embedding.
/// </summary>
internal static partial class TextPreprocessor
{
    /// <summary>
    /// Gets a normalized version of <paramref name="text"/> by lowercasing,
    /// removing non-alphanumeric characters (except spaces), and collapsing whitespace.
    /// </summary>
    /// <param name="text">The input text to clean.</param>
    /// <returns>
    /// The cleaned text. If <paramref name="text"/> is <c>null</c>, an empty string is returned.
    /// </returns>
    public static string Clean(string text)
    {
        if (text is null)
        {
            return string.Empty;
        }

        string t = text.ToLowerInvariant();
        t = NonAlnum().Replace(t, " ");
        t = MultiSpace().Replace(t, " ").Trim();
        return t;
    }

    [GeneratedRegex(@"[^a-z0-9 ]+", RegexOptions.Compiled)]
    private static partial Regex NonAlnum();

    [GeneratedRegex(@"\s+", RegexOptions.Compiled)]
    private static partial Regex MultiSpace();
}
