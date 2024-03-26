import os
import sys
import pytest

sys.path.append(os.getcwd() + '/')

if __name__ == '__main__':
    pytest.main(['-v', os.getcwd() + '/tests/'])