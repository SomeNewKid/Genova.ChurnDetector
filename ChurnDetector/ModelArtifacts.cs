// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Diagnostics.CodeAnalysis;

namespace Genova.ChurnDetector;

/// <summary>
/// Represents the deserialized runtime artifacts used by the detector,
/// including the model manifest and the label centroids.
/// </summary>
    [SuppressMessage("Style", "IDE0036:Order modifiers", Justification = "Conflicting order rules.")]
internal sealed class ModelArtifacts
{
    /// <summary>
    /// Gets the model manifest metadata (dimensions, thresholds, labels, and embedder info).
    /// </summary>
    required public Manifest Manifest { get; init; }

    /// <summary>
    /// Gets the collection of label-associated centroids used during similarity scoring.
    /// </summary>
    required public Centroid[] Centroids { get; init; }
}
