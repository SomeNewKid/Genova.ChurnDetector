// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Genova.ChurnDetector;

/// <summary>
/// Loads runtime artifacts (manifest and centroids) for the detector from embedded resources.
/// </summary>
internal static class ArtifactLoader
{
    /// <summary>
    /// Loads the <see cref="ModelArtifacts"/> (manifest and centroids) from the specified assembly's embedded resources.
    /// </summary>
    /// <param name="asm">The assembly that contains the embedded resources.</param>
    /// <returns>A populated <see cref="ModelArtifacts"/> instance.</returns>
    /// <exception cref="FileNotFoundError">Thrown when a required resource cannot be found.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the manifest or centroids are missing or invalid.</exception>
    public static ModelArtifacts Load(Assembly asm)
    {
        using Stream manifestStream = OpenResource(asm, Detector.ManifestResourceName);
        using Stream centroidStream = OpenResource(asm, Detector.CentroidsResourceName);

        using JsonDocument manDoc = JsonDocument.Parse(manifestStream);
        Manifest? manifest = JsonSerializer.Deserialize<Manifest>(manDoc.RootElement);

        using JsonDocument cenDoc = JsonDocument.Parse(centroidStream);
        Centroid[] centroids =
            JsonSerializer.Deserialize<CentroidsFile>(cenDoc.RootElement)?.Centroids
            ?? Array.Empty<Centroid>();

        if (manifest is null)
        {
            throw new InvalidOperationException("Manifest missing or invalid.");
        }

        if (centroids.Length == 0)
        {
            throw new InvalidOperationException("Centroids missing or invalid.");
        }

        return new ModelArtifacts
        {
            Manifest = manifest,
            Centroids = centroids,
        };
    }

    /// <summary>
    /// Opens an embedded resource stream by its file name within the given assembly.
    /// </summary>
    /// <param name="asm">The assembly to inspect for embedded resources.</param>
    /// <param name="fileName">The exact file name (e.g., "manifest-*.json") to locate.</param>
    /// <returns>A readable <see cref="Stream"/> for the embedded resource.</returns>
    /// <exception cref="FileNotFoundError">Thrown when the resource cannot be located.</exception>
    private static Stream OpenResource(Assembly asm, string fileName)
    {
        // Enumerate embedded resource names and find matches by suffix.
        string[] resourceNames = asm.GetManifestResourceNames();
        string[] candidates = resourceNames
            .Where(n => n.EndsWith("." + fileName, StringComparison.OrdinalIgnoreCase))
            .ToArray();

        if (candidates.Length > 0)
        {
            string candidate = candidates[0];
            Stream? stream = asm.GetManifestResourceStream(candidate);
            if (stream != null)
            {
                return stream;
            }
        }

        throw new FileNotFoundError($"Could not find '{fileName}' as an embedded resource.");
    }

    /// <summary>
    /// DTO for the centroids file payload.
    /// </summary>
    private sealed class CentroidsFile
    {
        /// <summary>
        /// Gets or sets the label-associated centroids used by the detector.
        /// </summary>
        [JsonPropertyName("centroids")]
        public Centroid[] Centroids { get; set; } = [];
    }

    /// <summary>
    /// Exception thrown when a required artifact cannot be found.
    /// </summary>
    private sealed class FileNotFoundError : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FileNotFoundError"/> class.
        /// </summary>
        /// <param name="message">The error message.</param>
        public FileNotFoundError(string message)
            : base(message)
        {
        }
    }
}
