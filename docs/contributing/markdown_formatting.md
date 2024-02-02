# Markdown Formatting

## Adding comments

If you want to add comments to your document that you don't want rendered to the VectorHub frontend, use the following
format in your markdown files. Make sure to create blank lines before and after your comment for the best results.

```markdown
[//]: # (your comment here)

or you use HTML comments

<!-- Your comment here -->
```

## Adding diagrams

You can use [mermaid](http://mermaid.js.org/intro/) to create diagrams for your content.

## Adding Special blocks in archbee

Archbee supports special code, tabs, link blocks, callouts, and changelog blocks which can be found in
[their documentation](https://docs.archbee.com/editor-markdown-shortcuts).

## Adding alt text and title to images

We encourage you to create alt text (for accessibility & SEO purposes) and a title (for explanability and readability)
for all images you add to a document.

```markdown
![Alt text](/path/to/img.jpg "Optional title")
```
