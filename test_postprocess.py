from postprocess import correct_sentence, word_error_rate


class TestCorrectSentence:
    def test_leaves_valid_sentence_unchanged(self):
        assert correct_sentence("bin blue at l six now") == "bin blue at l six now"

    def test_fixes_double_letter_collapse(self):
        assert correct_sentence("lay gren with t thre again") == "lay green with t three again"

    def test_fixes_soon_collapse(self):
        assert correct_sentence("bin white in fve son") == "bin white in five soon"

    def test_leaves_unmatchable_word_unchanged(self):
        assert correct_sentence("!!!!! blue at c four please") == "!!!!! blue at c four please"

    def test_empty_string(self):
        assert correct_sentence("") == ""


class TestWordErrorRate:
    def test_identical_sentences(self):
        assert word_error_rate("bin blue at l six now", "bin blue at l six now") == 0.0

    def test_completely_different_same_length(self):
        assert word_error_rate("a b c d e", "f g h i j") == 1.0

    def test_single_word_substitution(self):
        wer = word_error_rate("bin blue at l six now", "bin blue at l six soon")
        assert wer == 1 / 6

    def test_single_word_deletion(self):
        wer = word_error_rate("bin blue at l six now", "bin blue at l six")
        assert wer == 1 / 6

    def test_empty_hypothesis(self):
        assert word_error_rate("bin blue at l six now", "") == 1.0
