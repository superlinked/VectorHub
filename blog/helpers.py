import os
import enum

GIT_REPO = 'https://github.com/superlinked/VectorHub'

class ItemType(enum.Enum):
    FOLDER = "folder"
    FILE = "file"

class Item:
    def __init__(self, type, name, path, has_blogs=False, children=None):
        self.type = type
        self.name = name
        self.path = path
        self.has_blogs = has_blogs
        self.children = []

        if children:
            self.add_children(children)

    def __str__(self) -> str:
        return self.path

    def add_children(self, children):
        for child in children:
            # Recursively create DirectoryItem objects for children
            self.children.append(Item.from_dict(child))

    @classmethod
    def from_dict(cls, data):
        """
        Create an Item instance from a dictionary.
        """
        return cls(
            type=data.get("type", ItemType.FOLDER),
            name=data.get("name", ""),
            path=data.get("path", ""),
            has_blogs=data.get("has_blogs", False),
            children=data.get("children", [])
        )

    def to_dict(self):
        """
        Convert the Item object to a dictionary.
        """
        return {
            "type": self.type,
            "path": self.path,
            "name": self.name,
            "has_blogs": self.has_blogs,
            "children": self.children,
        }


class StrapiBlogType(enum.Enum):
    USECASE = "USECASE"
    ARTICLE = "ARTICLE"


class StrapiBlog:
    def __init__(self, content, filepath, last_updated, type: StrapiBlogType):
        self.content = content
        self.filepath = filepath
        self.last_updated = last_updated
        self.type = type

    def get_title(self) -> str:
        return os.path.basename(self.filepath).replace('-', ' ').replace('_', ' ').replace('.md', '')

    def __str__(self) -> str:
        return self.get_title()

    def get_github_url(self):
        return f'{GIT_REPO}/blob/main/{self.filepath}'
    
    
    def get_slug(self):
        return self.filepath.replace('.md', '').replace('_', '-').replace(' ', '-')

    def get_json(self):
        return {
        "github_url": self.get_github_url(),
        "content": self.content,
        "type": self.type.value,
        "github_last_updated_date": self.last_updated,
        "title": self.get_title(),
        "slug_url": self.get_slug()
    }

    def get_post_json(self):
        return {"data": self.get_json()}

    def __eq__(self, __value) -> bool:
        self.get_slug() == __value.get_slug()
