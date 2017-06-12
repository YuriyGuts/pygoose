import json
import pickle


def load(filename):
    """
    Deserialize a pickled object from disk.

    Args:
        filename: Path to the input pickle file.

    Returns:
        The deserialized object.
    """

    with open(filename, 'rb') as f:
        return pickle.load(f)


def save(obj, filename, protocol=4):
    """
    Serialize an object to disk using pickle protocol.

    Args:
        obj: The object to serialize.
        filename: Path to the output file.
        protocol: Version of the pickle protocol.
    """

    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def load_json(filename, **kwargs):
    """
    Load a JSON object from the specified file.

    Args:
        filename: Path to the input JSON file.
        **kwargs: Additional arguments to `json.load`.

    Returns:
        The object deserialized from JSON.
    """

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f, **kwargs)


def save_json(obj, filename, **kwargs):
    """
    Save an object as a JSON file.

    Args:
        obj: The object to save. Must be JSON-serializable.
        filename: Path to the output file.
        **kwargs: Additional arguments to `json.dump`.
    """

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, **kwargs)


def load_lines(filename):
    """
    Load a text file as an array of lines.

    Args:
        filename: Path to the input file.

    Returns:
        An array of strings, each representing an individual line.
    """

    with open(filename, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f.readlines()]


def save_lines(lines, filename):
    """
    Save an array of lines to a file.

    Args:
        lines: An array of strings that will be saved as individual lines.
        filename: Path to the output file.
    """

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
