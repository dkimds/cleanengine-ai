"""
General chain for handling miscellaneous queries.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any


class GeneralChain:
    """
    Handles general queries that don't fit into specific categories.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the general chain.
        
        Args:
            model: OpenAI model to use for response generation
        """
        self.model = model
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the general chain with prompt and model."""
        
        # General prompt template
        self.prompt = PromptTemplate.from_template(
            """Previous conversation:
{chat_history}

Respond to the following question concisely:
If the question is not about expert knowledge or recent events, reply:

"도와드리지 못해서 죄송합니다. 저는 비트코인 관련 전문지식과 최신소식만 답변드릴 수 있습니다."

Only respond with factual, concise answers supported by the context when applicable.
Question: {question}
Answer:
"""
        )
        
        # Create the chain
        self.chain = (
            self.prompt
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
        )
    
    def process(self, question: str, chat_history: str = "") -> str:
        """
        Process a general question.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            General response or polite refusal
        """
        return self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invoke the general chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            General response as string
        """
        return self.chain.invoke(inputs)
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the general chain.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            General response as string
        """
        return await self.chain.ainvoke(inputs)
    
    def get_chain(self):
        """Get the underlying chain object."""
        return self.chain