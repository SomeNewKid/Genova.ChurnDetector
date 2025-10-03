

namespace Genova.ChurnDetector;

internal sealed class ModelArtifacts
{
    public required Manifest Manifest { get; init; }
    public required Centroid[] Centroids { get; init; }
}
