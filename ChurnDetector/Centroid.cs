
namespace Genova.ChurnDetector;

internal sealed class Centroid
{
    public string label { get; set; } = "";
    public float[] vector { get; set; } = new float[0];
}
