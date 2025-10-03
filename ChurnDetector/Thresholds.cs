// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Text.Json.Serialization;

namespace Genova.ChurnDetector;

/// <summary>
/// Defines decision thresholds applied during churn scoring.
/// </summary>
internal sealed class Thresholds
{
    /// <summary>
    /// Gets or sets the minimum cosine similarity required to accept a churn prediction.
    /// </summary>
    [JsonPropertyName("churn_min_similarity")]
    public float ChurnMinSimilarity { get; set; } = 0.75f;

    /// <summary>
    /// Gets or sets the minimum margin required between the top-1 and top-2 label scores.
    /// </summary>
    [JsonPropertyName("min_margin")]
    public float MinMargin { get; set; } = 0.05f;
}
