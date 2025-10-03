

namespace Genova.ChurnDetector;

internal sealed class Manifest
{
    public int version { get; set; }
    public string? created_utc { get; set; }
    public int vector_dim { get; set; }
    public string[] labels { get; set; } = new string[0];
    public Thresholds thresholds { get; set; } = new Thresholds();
    public EmbedderInfo embedder { get; set; } = new EmbedderInfo();
}
