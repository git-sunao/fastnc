import io
import logging
import unittest

import fastnc
from fastnc._logging import log


class LoggingTests(unittest.TestCase):
    def tearDown(self):
        fastnc.disable_logging()
        logging.getLogger("fastnc").setLevel(logging.NOTSET)

    def test_configure_logging_controls_verbosity_without_duplicate_handlers(self):
        stream = io.StringIO()
        logger = fastnc.configure_logging("INFO", stream=stream)
        fastnc.configure_logging("INFO", stream=stream)

        log(logging.getLogger("fastnc.test"), logging.DEBUG, "hidden")
        log(logging.getLogger("fastnc.test"), logging.INFO, "visible stage")

        output = stream.getvalue()
        self.assertNotIn("hidden", output)
        self.assertEqual(output.count("visible stage"), 1)
        self.assertEqual(logger.level, logging.INFO)

    def test_debug_level_exposes_diagnostics(self):
        stream = io.StringIO()
        fastnc.configure_logging(logging.DEBUG, stream=stream)

        log(logging.getLogger("fastnc.test"), logging.DEBUG, "cache hit")

        self.assertIn("DEBUG fastnc: cache hit", stream.getvalue())

    def test_unknown_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown logging level"):
            fastnc.configure_logging("verbose")


if __name__ == "__main__":
    unittest.main()
