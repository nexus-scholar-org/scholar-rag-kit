import re
from typing import List, Dict, Any
from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    metadata: Dict[str, Any]

class MarkdownChunker:
    @staticmethod
    def chunk(markdown_text: str, base_metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Splits a markdown document by its headers, preserving structural context.
        """
        if base_metadata is None:
            base_metadata = {}
            
        lines = markdown_text.split('\n')
        chunks = []
        
        current_section = "Abstract/Intro"
        current_text = []
        
        header_pattern = re.compile(r'^(#+)\s+(.*)$')
        
        for line in lines:
            match = header_pattern.match(line)
            if match:
                # Save previous chunk if it has content
                if any(t.strip() for t in current_text):
                    meta = base_metadata.copy()
                    meta["section"] = current_section
                    chunks.append(Chunk(text="\n".join(current_text).strip(), metadata=meta))
                    
                # Start new chunk
                current_section = match.group(2).strip()
                current_text = [line]
            else:
                current_text.append(line)
                
        # Save last chunk
        if any(t.strip() for t in current_text):
            meta = base_metadata.copy()
            meta["section"] = current_section
            chunks.append(Chunk(text="\n".join(current_text).strip(), metadata=meta))
            
        return chunks
