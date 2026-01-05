# Paper Writing Guidelines

## AI-Generated Content

All AI-generated edits, suggestions, or drafts must be marked with the `\ai{}` command:

```latex
\ai{This text was written or suggested by AI}
```

This renders as green text: [AI: ...] and ensures transparency about which content needs human review.

## Author Comment Commands

```latex
\joey{comment}   % orange
\itamar{comment} % blue
\core{comment}   % red
\ai{comment}     % green
```

## Workflow

1. AI suggests edits marked with `\ai{}`
2. Human reviews and either accepts (removing `\ai{}`) or revises
3. Final submission should have no `\ai{}` markers remaining
