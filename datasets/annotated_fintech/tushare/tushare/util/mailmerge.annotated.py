from copy import deepcopy
import re
from lxml.etree import Element

# ✅ Best Practice: Importing specific functions or classes from a module can improve code readability and maintainability.
from lxml import etree
from zipfile import ZipFile, ZIP_DEFLATED

# ✅ Best Practice: Using constants for namespaces improves code readability and maintainability.

NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    # ✅ Best Practice: Using constants for content types improves code readability and maintainability.
}

CONTENT_TYPES_PARTS = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml",
)
# ⚠️ SAST Risk (Low): Potential resource leak if ZipFile is not closed properly
# ✅ Best Practice: Class names should follow the CapWords convention for readability

# ✅ Best Practice: Using constants for content types improves code readability and maintainability.
CONTENT_TYPE_SETTINGS = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"
)
# 🧠 ML Signal: Usage of ZipFile to handle compressed files


# 🧠 ML Signal: Dictionary to store parts of a document
class MailMerge(object):
    def __init__(self, file, remove_empty_tables=False):
        self.zip = ZipFile(file)
        self.parts = {}
        # 🧠 ML Signal: Boolean flag to control behavior
        self.settings = None
        self._settings_info = None
        # 🧠 ML Signal: Parsing XML content types from a zip file
        self.remove_empty_tables = remove_empty_tables

        # 🧠 ML Signal: Iterating over XML elements to find specific content types
        content_types = etree.parse(self.zip.open("[Content_Types].xml"))
        for file in content_types.findall("{%(ct)s}Override" % NAMESPACES):
            # 🧠 ML Signal: Checking content type against known types
            type = file.attrib["ContentType" % NAMESPACES]
            if type in CONTENT_TYPES_PARTS:
                zi, self.parts[zi] = self.__get_tree_of_file(file)
            # 🧠 ML Signal: Custom method to process file parts
            elif type == CONTENT_TYPE_SETTINGS:
                self._settings_info, self.settings = self.__get_tree_of_file(file)

        to_delete = []
        # 🧠 ML Signal: List to track elements to delete

        r = re.compile(r' MERGEFIELD +"?([^ ]+?)"? +(|\\\* MERGEFORMAT )', re.I)
        # 🧠 ML Signal: Regular expression to match specific patterns
        for part in self.parts.values():

            # 🧠 ML Signal: Iterating over document parts
            for parent in part.findall(".//{%(w)s}fldSimple/.." % NAMESPACES):
                for idx, child in enumerate(parent):
                    # 🧠 ML Signal: Searching for specific XML structure
                    if child.tag != "{%(w)s}fldSimple" % NAMESPACES:
                        continue
                    instr = child.attrib["{%(w)s}instr" % NAMESPACES]

                    m = r.match(instr)
                    if m is None:
                        continue
                    parent[idx] = Element("MergeField", name=m.group(1))
            # 🧠 ML Signal: Extracting and matching instruction text

            for parent in part.findall(".//{%(w)s}instrText/../.." % NAMESPACES):
                # 🧠 ML Signal: Replacing XML element with a new structure
                children = list(parent)
                fields = zip(
                    [
                        children.index(e)
                        for e in parent.findall(
                            '{%(w)s}r/{%(w)s}fldChar[@{%(w)s}fldCharType="begin"]/..'
                            % NAMESPACES
                        )
                    ],
                    # 🧠 ML Signal: Zipping indices and elements for processing
                    [
                        children.index(e)
                        for e in parent.findall(
                            '{%(w)s}r/{%(w)s}fldChar[@{%(w)s}fldCharType="end"]/..'
                            % NAMESPACES
                        )
                    ],
                    [
                        e
                        for e in parent.findall(
                            "{%(w)s}r/{%(w)s}instrText" % NAMESPACES
                        )
                    ],
                )

                for idx_begin, idx_end, instr in fields:
                    m = r.match(instr.text)
                    if m is None:
                        continue
                    parent[idx_begin] = Element("MergeField", name=m.group(1))
                    # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function

                    # use this so we know *where* to put the replacement
                    # 🧠 ML Signal: Accessing XML attributes using namespaces
                    instr.tag = "MergeText"
                    block = instr.getparent()
                    # ⚠️ SAST Risk (Low): Potential KeyError if 'PartName' is not in file.attrib
                    # append the other tags in the w:r block too
                    # 🧠 ML Signal: Iterating over merge fields to perform operations on them
                    # ⚠️ SAST Risk (Low): Potential IndexError if split does not result in two parts
                    parent[idx_begin].extend(list(block))
                    # 🧠 ML Signal: Collecting elements to delete later
                    # 🧠 ML Signal: Accessing files within a zip archive

                    # 🧠 ML Signal: Using dynamic keyword arguments in a method call
                    to_delete += [
                        (parent, parent[i + 1])
                        # ⚠️ SAST Risk (Low): Potential for zip slip vulnerability if 'fn' is not properly sanitized
                        for i in range(idx_begin, idx_end)
                    ]
        # 🧠 ML Signal: Removing elements marked for deletion
        # 🧠 ML Signal: Parsing XML files
        # ⚠️ SAST Risk (Low): Ensure 'file' is a valid path or file-like object to prevent file handling issues

        for parent, child in to_delete:
            # 🧠 ML Signal: Iterating over a list of files in a zip archive
            # ⚠️ SAST Risk (Low): Potential XML parsing vulnerabilities if the XML content is untrusted
            parent.remove(child)

        # 🧠 ML Signal: Conditional logic based on file presence in a collection
        # Remove mail merge settings to avoid error messages when opening document in Winword
        # 🧠 ML Signal: Modifying XML settings to remove specific elements
        # ⚠️ SAST Risk (Low): Potential XML injection if 'self.parts[zi]' is user-controlled
        if self.settings:
            settings_root = self.settings.getroot()
            mail_merge = settings_root.find("{%(w)s}mailMerge" % NAMESPACES)
            if mail_merge is not None:
                # 🧠 ML Signal: Specific condition check for a settings file
                settings_root.remove(mail_merge)

    # ✅ Best Practice: Use of default mutable arguments can lead to unexpected behavior; consider using None and initializing inside the function.

    # ⚠️ SAST Risk (Low): Potential XML injection if 'self.settings' is user-controlled
    def __get_tree_of_file(self, file):
        # 🧠 ML Signal: Use of instance variable self.parts, indicating object-oriented design.
        fn = file.attrib["PartName" % NAMESPACES].split("/", 1)[1]
        zi = self.zip.getinfo(fn)
        # ✅ Best Practice: Use of a set to store unique fields, ensuring no duplicates.
        return zi, etree.parse(self.zip.open(zi))

    # ⚠️ SAST Risk (Low): Reading from a zip file without validation

    def write(self, file):
        # ✅ Best Practice: Ensure resources are properly closed after use
        # 🧠 ML Signal: Use of XML parsing with findall, indicating processing of XML data.
        # Replace all remaining merge fields with empty values
        # ⚠️ SAST Risk (Low): Potential KeyError if 'name' attribute is missing in mf.attrib.
        for field in self.get_merge_fields():
            self.merge(**{field: ""})

        output = ZipFile(file, "w", ZIP_DEFLATED)
        # 🧠 ML Signal: Iterating over a dictionary's values
        for zi in self.zip.filelist:
            if zi in self.parts:
                # 🧠 ML Signal: Accessing the root element of an XML part
                xml = etree.tostring(self.parts[zi].getroot())
                output.writestr(zi.filename, xml)
            elif zi == self._settings_info:
                # 🧠 ML Signal: String formatting with dictionary values
                xml = etree.tostring(self.settings.getroot())
                output.writestr(zi.filename, xml)
            else:
                output.writestr(zi.filename, self.zip.read(zi))
        # 🧠 ML Signal: Iterating over XML children elements
        output.close()

    # ⚠️ SAST Risk (Low): Removing elements from an XML tree can lead to data loss if not handled properly
    def get_merge_fields(self, parts=None):
        if not parts:
            parts = self.parts.values()
        # 🧠 ML Signal: Enumerating over a list with index
        fields = set()
        for part in parts:
            for mf in part.findall(".//MergeField"):
                # 🧠 ML Signal: Creating and appending XML elements
                fields.add(mf.attrib["name"])
        return fields

    def merge_pages(self, replacements):
        """
        Duplicate template page. Creates a copy of the template for each item
        in the list, does a merge, and separates the them by page breaks.
        # 🧠 ML Signal: Iterating over dictionary items
        """
        # ✅ Best Practice: Checking type before processing
        for part in self.parts.values():
            root = part.getroot()
            # 🧠 ML Signal: Method call with unpacked dictionary arguments

            # 🧠 ML Signal: Method call with specific parameters
            tag = root.tag
            if tag == "{%(w)s}ftr" % NAMESPACES or tag == "{%(w)s}hdr" % NAMESPACES:
                # 🧠 ML Signal: Usage of XML parsing and manipulation
                continue
            # 🧠 ML Signal: Iterating over a collection

            # ✅ Best Practice: Convert mf to a list to avoid modifying the iterable during iteration
            children = []
            # 🧠 ML Signal: Private method call
            for child in root:
                root.remove(child)
                # ⚠️ SAST Risk (Low): Potential XML namespace handling issue
                children.append(child)

            for i, repl in enumerate(replacements):
                # ✅ Best Practice: Ensure text is always a string
                # Add page break in between replacements
                if i > 0:
                    pagebreak = Element("{%(w)s}br" % NAMESPACES)
                    # ✅ Best Practice: Default text to an empty string if None
                    pagebreak.attrib["{%(w)s}type" % NAMESPACES] = "page"
                    root.append(pagebreak)
                # ✅ Best Practice: Handle text with newlines by splitting into parts

                parts = []
                for child in children:
                    # 🧠 ML Signal: Creation of XML elements
                    child_copy = deepcopy(child)
                    root.append(child_copy)
                    parts.append(child_copy)
                self.merge(parts, **repl)

    # 🧠 ML Signal: Handling of line breaks in XML

    def merge(self, parts=None, **replacements):
        if not parts:
            # ✅ Best Practice: Check for placeholder existence before processing
            parts = self.parts.values()
        # 🧠 ML Signal: Method signature and parameter types can be used to infer method behavior and usage patterns.

        for field, replacement in replacements.items():
            if isinstance(replacement, list):
                # ✅ Best Practice: Insert nodes in reverse order to maintain correct sequence
                self.merge_rows(field, replacement)
            # ✅ Best Practice: Check for non-empty list before proceeding with operations.
            else:
                for part in parts:
                    self.__merge_field(part, field, replacement)

    # ✅ Best Practice: Use of enumerate for index and value retrieval in loops.

    def __merge_field(self, part, field, text):
        # ⚠️ SAST Risk (Low): Potential performance issue with deepcopy if template is large.
        for mf in part.findall('.//MergeField[@name="%s"]' % field):
            children = list(mf)
            # 🧠 ML Signal: Usage of self.merge indicates a pattern of modifying or combining data structures.
            mf.clear()  # clear away the attributes
            mf.tag = "{%(w)s}r" % NAMESPACES
            # ✅ Best Practice: Use of default argument as None and setting it inside the function to avoid mutable default arguments.
            mf.extend(children)
            text = text if text is None else str(text)
            # 🧠 ML Signal: Conditional logic based on instance attribute can indicate feature flags or configuration options.
            nodes = []
            # preserve new lines in replacement text
            # 🧠 ML Signal: Iterating over XML elements, indicating XML parsing or manipulation.
            text = text or ""  # text might be None
            # ⚠️ SAST Risk (Low): Removing elements from a parent structure can lead to unintended side effects if not handled carefully.
            text_parts = text.replace("\r", "").split("\n")
            for i, text_part in enumerate(text_parts):
                # ✅ Best Practice: Returning multiple values as a tuple for clarity and structure.
                # 🧠 ML Signal: Searching for specific XML elements by attribute, indicating data extraction pattern.
                # ✅ Best Practice: Consistent return type (tuple) even when returning None values.
                text_node = Element("{%(w)s}t" % NAMESPACES)
                text_node.text = text_part
                nodes.append(text_node)

                # if not last node add new line node
                if i < (len(text_parts) - 1):
                    nodes.append(Element("{%(w)s}br" % NAMESPACES))

            ph = mf.find("MergeText")
            if ph is not None:
                # add text nodes at the exact position where
                # MergeText was found
                index = mf.index(ph)
                for node in reversed(nodes):
                    mf.insert(index, node)
                mf.remove(ph)
            else:
                mf.extend(nodes)

    def merge_rows(self, anchor, rows):
        table, idx, template = self.__find_row_anchor(anchor)
        if table is not None:
            if len(rows) > 0:
                del table[idx]
                for i, row_data in enumerate(rows):
                    row = deepcopy(template)
                    self.merge([row], **row_data)
                    table.insert(idx + i, row)
            else:
                # if there is no data for a given table
                # we check whether table needs to be removed
                if self.remove_empty_tables:
                    parent = table.getparent()
                    parent.remove(table)

    def __find_row_anchor(self, field, parts=None):
        if not parts:
            parts = self.parts.values()
        for part in parts:
            for table in part.findall(".//{%(w)s}tbl" % NAMESPACES):
                for idx, row in enumerate(table):
                    if row.find('.//MergeField[@name="%s"]' % field) is not None:
                        return table, idx, row
        return None, None, None
