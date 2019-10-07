import logging
import os
import shutil

from download import download

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_download():
    logger.info('start test_download function')

    # Base URL for downloading the data-files from the internet.
    url = 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf'

    # prepare target dir to save the downloaded file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(root_dir, 'test')

    # remove the dir if exists
    logger.info('Remove dir: %s', target_dir)
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        shutil.rmtree(target_dir)

    logger.info('The saved dir: %s', target_dir)
    file_path = download(url, target_dir)

    expected_file_name = os.path.join(target_dir, 'dummy.pdf')
    assert file_path == expected_file_name

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     test_download()
