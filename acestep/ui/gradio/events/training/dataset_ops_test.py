"""Unit tests for dataset_ops.py."""

import unittest
from unittest.mock import MagicMock, patch

from acestep.ui.gradio.events.training.dataset_ops import (
    auto_label_all,
    get_sample_preview,
    save_sample_edit,
    update_settings,
    save_dataset,
)


class TestGetSamplePreview(unittest.TestCase):
    """Tests for get_sample_preview."""

    def test_none_builder_returns_empty(self):
        result = get_sample_preview(0, None)
        # Should return the empty tuple
        self.assertIsNone(result[0])  # audio_path
        self.assertEqual(result[1], "")  # filename

    def test_empty_samples_returns_empty(self):
        builder = MagicMock()
        builder.samples = []
        result = get_sample_preview(0, builder)
        self.assertIsNone(result[0])

    def test_none_index_returns_empty(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        result = get_sample_preview(None, builder)
        self.assertIsNone(result[0])

    def test_out_of_range_index_returns_empty(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        result = get_sample_preview(5, builder)
        self.assertIsNone(result[0])

    def test_valid_sample_returns_data(self):
        sample = MagicMock()
        sample.audio_path = "/path/to/audio.wav"
        sample.filename = "audio.wav"
        sample.caption = "Test caption"
        sample.genre = "rock"
        sample.prompt_override = "genre"
        sample.lyrics = "Hello world"
        sample.formatted_lyrics = ""
        sample.bpm = 120
        sample.keyscale = "C major"
        sample.timesignature = "4/4"
        sample.duration = 30.0
        sample.language = "en"
        sample.is_instrumental = False
        sample.raw_lyrics = ""
        sample.has_raw_lyrics.return_value = False

        builder = MagicMock()
        builder.samples = [sample]

        result = get_sample_preview(0, builder)
        self.assertEqual(result[0], "/path/to/audio.wav")
        self.assertEqual(result[1], "audio.wav")
        self.assertEqual(result[4], "Genre")  # prompt_override converted


class TestUpdateSettings(unittest.TestCase):
    """Tests for update_settings."""

    def test_none_builder_returns_none(self):
        result = update_settings("tag", "prefix", False, 50, None)
        self.assertIsNone(result)

    def test_updates_genre_ratio(self):
        builder = MagicMock()
        builder.metadata = MagicMock()
        result = update_settings("", "prefix", False, 75, builder)
        self.assertEqual(result.metadata.genre_ratio, 75)


class TestSaveDataset(unittest.TestCase):
    """Tests for save_dataset."""

    def test_none_builder(self):
        status, _ = save_dataset("path.json", "name", None)
        self.assertIn("❌", status)

    def test_empty_samples(self):
        builder = MagicMock()
        builder.samples = []
        status, _ = save_dataset("path.json", "name", builder)
        self.assertIn("❌", status)

    def test_empty_path(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        status, _ = save_dataset("", "name", builder)
        self.assertIn("❌", status)


class TestAutoLabelAll(unittest.TestCase):
    """Tests for auto_label_all lazy-initialization behavior."""

    @patch("acestep.ui.gradio.events.training.dataset_ops.gr.update", side_effect=lambda **kwargs: kwargs)
    @patch(
        "acestep.ui.gradio.events.training.dataset_ops.ensure_training_labeling_model_ready",
        return_value=("✅ Model initialized", True),
    )
    @patch(
        "acestep.ui.gradio.events.training.dataset_ops.ensure_training_labeling_llm_ready",
        return_value=("✅ LLM initialized", True),
    )
    def test_initializes_model_and_llm_on_demand_before_labeling(
        self,
        mock_ensure_llm_ready,
        mock_ensure_model_ready,
        _mock_gr_update,
    ):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        builder.get_samples_dataframe_data.return_value = [["sample.wav", "caption"]]
        builder.label_all_samples.return_value = (builder.samples, "✅ labeled")

        dit_handler = MagicMock()
        dit_handler.model = None
        llm_handler = MagicMock()
        llm_handler.llm_initialized = False

        table_update, status_update, returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
        )

        mock_ensure_model_ready.assert_called_once_with(dit_handler)
        mock_ensure_llm_ready.assert_called_once_with(llm_handler)
        builder.label_all_samples.assert_called_once()
        self.assertEqual([["sample.wav", "caption"]], table_update["value"])
        self.assertEqual("✅ labeled", status_update["value"])
        self.assertIs(returned_builder, builder)

    @patch("acestep.ui.gradio.events.training.dataset_ops.gr.update", side_effect=lambda **kwargs: kwargs)
    @patch(
        "acestep.ui.gradio.events.training.dataset_ops.ensure_training_labeling_model_ready",
        return_value=("❌ Model init failed", False),
    )
    def test_returns_model_init_error_when_lazy_initialization_fails(
        self,
        mock_ensure_model_ready,
        _mock_gr_update,
    ):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        builder.get_samples_dataframe_data.return_value = [["sample.wav", "caption"]]

        dit_handler = MagicMock()
        dit_handler.model = None
        llm_handler = MagicMock()
        llm_handler.llm_initialized = True

        table_update, status_update, returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
        )

        mock_ensure_model_ready.assert_called_once_with(dit_handler)
        builder.label_all_samples.assert_not_called()
        self.assertEqual([["sample.wav", "caption"]], table_update)
        self.assertEqual("❌ Model init failed", status_update)
        self.assertIs(returned_builder, builder)

    @patch("acestep.ui.gradio.events.training.dataset_ops.gr.update", side_effect=lambda **kwargs: kwargs)
    @patch(
        "acestep.ui.gradio.events.training.dataset_ops.ensure_training_labeling_model_ready",
        return_value=("✅ Model initialized", True),
    )
    @patch(
        "acestep.ui.gradio.events.training.dataset_ops.ensure_training_labeling_llm_ready",
        return_value=("❌ LLM init failed", False),
    )
    def test_returns_llm_init_error_when_lazy_initialization_fails(
        self,
        mock_ensure_llm_ready,
        _mock_ensure_model_ready,
        _mock_gr_update,
    ):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        builder.get_samples_dataframe_data.return_value = [["sample.wav", "caption"]]

        dit_handler = MagicMock()
        dit_handler.model = None
        llm_handler = MagicMock()
        llm_handler.llm_initialized = False

        table_update, status_update, returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
        )

        mock_ensure_llm_ready.assert_called_once_with(llm_handler)
        builder.label_all_samples.assert_not_called()
        self.assertEqual([["sample.wav", "caption"]], table_update)
        self.assertEqual("❌ LLM init failed", status_update)
        self.assertIs(returned_builder, builder)


if __name__ == "__main__":
    unittest.main()
