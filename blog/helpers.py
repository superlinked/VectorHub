import enum
import json

class ItemType(enum.Enum):
    FOLDER = "folder"
    FILE = "file"

class Item:
    def __init__(self, type, name, path, children):
        self.type = type
        self.name = name
        self.path = path
        self.children = children

    @classmethod
    def from_dict(cls, data):
        """
        Create an Item instance from a dictionary.
        """
        return cls(
            type=data.get("type", ItemType.FILE),
            name=data.get("name", ""),
            path=data.get("path", ""),
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
            "children": self.children,
        }


class Items:
    def __init__(self):
        self.items = []

    def load_from_file(self, filename):
        """
        Load item data from a JSON file.
        """
        with open(filename, 'r') as file:
            data = json.load(file)
            for item_data in data:
                self.items.append(Item.from_dict(item_data))

    def add_item(self, user):
        """
        Add an Item object to the list.
        """
        if isinstance(user, Item):
            self.items.append(iter)
        else:
            raise TypeError("item must be an instance of Item")

    def to_json(self):
        """
        Convert the Items object back to a JSON string.
        """
        return json.dumps([item.to_dict() for item in self.items], indent=4)