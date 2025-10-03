// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Text.Json.Serialization;

namespace Genova.ChurnDetector;

/// <summary>
/// Represents a labeled centroid used by the churn detector during similarity scoring.
/// </summary>
internal sealed class Centroid
{
    /// <summary>
    /// Gets or sets the label associated with this centroid (e.g., "churn", "keep", or "unsure").
    /// </summary>
    [JsonPropertyName("label")]
    public string Label { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the embedding-space vector that defines the centroid.
    /// </summary>
    [JsonPropertyName("vector")]
    public float[] Vector { get; set; } = [];
}
