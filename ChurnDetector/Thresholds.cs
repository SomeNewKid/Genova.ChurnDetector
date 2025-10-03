

namespace Genova.ChurnDetector;

internal sealed class Thresholds
{
    public float churn_min_similarity { get; set; } = 0.75f;
    public float min_margin { get; set; } = 0.05f;
}
