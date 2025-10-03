// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using FluentAssertions;

namespace Genova.ChurnDetector.UnitTests;

/// <summary>
/// Unit tests for <see cref="TextPreprocessor"/>.
/// </summary>
public sealed class TextPreprocessor_Tests
{
    [Fact]
    public void Clean_when_null_returns_empty_string()
    {
        // Act
        string result = TextPreprocessor.Clean(null!);

        // Assert
        result.Should().BeEmpty();
    }

    [Theory]
    [InlineData(" Hello, WORLD!!  123\t\t", "hello world 123")]
    [InlineData("A_B-C+D", "a b c d")]
    [InlineData("  Keep   CAPS & Punct.!@#   ", "keep caps punct")]
    public void Clean_normalizes_text(string input, string expected)
    {
        // Act
        string result = TextPreprocessor.Clean(input);

        // Assert
        result.Should().Be(expected);
    }
}
