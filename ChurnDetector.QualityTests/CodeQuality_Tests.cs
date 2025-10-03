// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using Genova.Testing.QualityTests;

namespace Genova.ChurnDetector.QualityTests;

[TestClass]
public class CodeQuality_Tests : CodeQuality_Base
{
    public CodeQuality_Tests()
        : base(typeof(Detector).Assembly, "Genova.ChurnDetector")
    {
    }
}
