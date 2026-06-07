// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

namespace Genova.ChurnDetector;

/// <summary>
/// Represents the deserialized runtime artifacts used by the detector,
/// including the model manifest and the label centroids.
/// </summary>
internal sealed class ModelArtifacts
{
    /// <summary>
    /// Gets the model manifest metadata (dimensions, thresholds, labels, and embedder info).
    /// </summary>
    public required Manifest Manifest { get; init; }

    /// <summary>
    /// Gets the collection of label-associated centroids used during similarity scoring.
    /// </summary>
    public required Centroid[] Centroids { get; init; }
}
