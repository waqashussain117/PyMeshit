"""Choose the next patch tag, reusing an existing tag when retrying a commit."""

import re
import subprocess


def next_release(all_tags, commit_tags):
    def versions(tags):
        for tag in tags:
            match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
            if match:
                yield tuple(map(int, match.groups()))

    existing = list(versions(commit_tags))
    if existing:
        return "v{}.{}.{}".format(*max(existing))
    latest = max(versions(all_tags), default=(0, 0, 0))
    next_version = max((1, 0, 0), (latest[0], latest[1], latest[2] + 1))
    return "v{}.{}.{}".format(*next_version)


if __name__ == "__main__":
    all_tags = subprocess.check_output(["git", "tag", "--list"], text=True).splitlines()
    commit_tags = subprocess.check_output(["git", "tag", "--points-at", "HEAD"], text=True).splitlines()
    print(next_release(all_tags, commit_tags))
