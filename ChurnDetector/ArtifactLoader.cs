
using System.Reflection;
using System.Text.Json;

namespace Genova.ChurnDetector;

internal static class ArtifactLoader
{
    public static ModelArtifacts Load(Assembly asm)
    {
        using var manifestStream = OpenResource(asm, Detector.ManifestResourceName);
        if (manifestStream == null) throw new FileNotFoundError("Manifest not found");

        using var centroidStream = OpenResource(asm, Detector.CentroidsResourceName);
        if (centroidStream == null) throw new FileNotFoundError("Centroids not found");

        using var manDoc = JsonDocument.Parse(manifestStream);
        var manifest = JsonSerializer.Deserialize<Manifest>(manDoc.RootElement);

        using var cenDoc = JsonDocument.Parse(centroidStream);
        var centroids = JsonSerializer.Deserialize<CentroidsFile>(cenDoc.RootElement)?.centroids ?? [];

        if (manifest is null) throw new InvalidOperationException("Manifest missing");
        if (centroids.Length == 0) throw new InvalidOperationException("Centroids missing");

        return new ModelArtifacts { Manifest = manifest, Centroids = centroids };
    }

    private static Stream OpenResource(Assembly asm, string fileName)
    {
        // Try embedded resource first
        var candidates = asm.GetManifestResourceNames()
                            .Where(n => n.EndsWith("." + fileName, StringComparison.OrdinalIgnoreCase))
                            .ToArray();
        if (candidates.Length > 0)
        {
            string candidate = candidates[0];
            var stream = asm.GetManifestResourceStream(candidate);
            if (stream != null)
            {
                return stream;
            }
        }

        throw new FileNotFoundError($"Could not find '{fileName}' as embedded resource or in ./Data");
    }

    private sealed class CentroidsFile
    {
        public Centroid[] centroids { get; set; } = new Centroid[0];
    }

    private sealed class FileNotFoundError : Exception
    {
        public FileNotFoundError(string message) : base(message) { }
    }
}
