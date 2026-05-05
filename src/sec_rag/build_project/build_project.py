from sec_rag.build.project_structure import DIRS


def build():
    for path in DIRS:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    build()
    print("Project directories ready.")