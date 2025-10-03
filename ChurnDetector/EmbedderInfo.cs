
namespace Genova.ChurnDetector;

internal sealed class EmbedderInfo
{
    public string type { get; set; } = "CharTrigramHashing";
    public int ngram { get; set; } = 3;
    public int dim { get; set; } = 256;
    public string hash { get; set; } = "FNV1A64";
    public bool signed { get; set; } = true;
    public bool lowercase { get; set; } = true;
    public bool keep_alnum_space_only { get; set; } = true;
    public int padding_spaces { get; set; } = 2;
}
