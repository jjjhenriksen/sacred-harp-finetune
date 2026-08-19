from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssociatedTune:
    book: str
    page: str
    tune: str
    edition: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        data = {
            "book": self.book,
            "page": self.page,
            "tune": self.tune,
        }
        if self.edition:
            data["edition"] = self.edition
        if self.url:
            data["url"] = self.url
        return data


@dataclass(frozen=True)
class StanzaRecord:
    id: str
    text_id: str
    incipit: str
    first_line: str
    text: str
    associated_tunes: list[AssociatedTune] = field(default_factory=list)
    associated_songs: list[str] = field(default_factory=list)
    source_books: list[str] = field(default_factory=list)
    theme_tags: list[str] = field(default_factory=list)
    tone_tags: list[str] = field(default_factory=list)
    operational_fit: list[str] = field(default_factory=list)
    tag_rationale: str = ""


@dataclass(frozen=True)
class UsageEntry:
    stanza_id: str
    date_used: str


@dataclass(frozen=True)
class EnrichedStanza:
    id: str
    text_id: str
    incipit: str
    first_line: str
    text: str
    associated_tunes: list[AssociatedTune]
    associated_songs: list[str]
    source_books: list[str]
    theme_tags: list[str]
    tone_tags: list[str]
    operational_fit: list[str]
    tag_rationale: str
    last_used_date: str | None
    times_used: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text_id": self.text_id,
            "incipit": self.incipit,
            "first_line": self.first_line,
            "text": self.text,
            "associated_tunes": [item.to_dict() for item in self.associated_tunes],
            "associated_songs": list(self.associated_songs),
            "source_books": list(self.source_books),
            "theme_tags": list(self.theme_tags),
            "tone_tags": list(self.tone_tags),
            "operational_fit": list(self.operational_fit),
            "tag_rationale": self.tag_rationale,
            "last_used_date": self.last_used_date,
            "times_used": self.times_used,
        }
