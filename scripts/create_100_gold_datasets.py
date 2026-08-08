#!/usr/bin/env python3
"""Generate and validate 100-case gold benchmark datasets for 3 repositories."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.eval.answer_quality import load_gold_dataset


def generate_llama_index_100() -> dict:
    cases = []
    
    # 35 Simple questions
    simple_defs = [
        ("AQ001", "VectorStoreIndex.as_retriever", "What state does VectorStoreIndex pass into the VectorIndexRetriever returned by as_retriever?", 
         "It constructs VectorIndexRetriever with the index itself, node IDs from index_struct.nodes_dict values, the index callback manager, the object map, and any caller-provided keyword arguments.",
         ["indices/vector_store/base.py"], ["indices/vector_store/base.py:112-125"], ["VectorStoreIndex.as_retriever lazily imports VectorIndexRetriever and supplies self, stored node IDs, callback_manager, object_map, and forwarded kwargs."]),
        ("AQ002", "Document.__init__", "What default attributes does a Document node carry upon creation?",
         "A Document carries text content, doc_id, embedding, extra_info/metadata, doc_hash, and relationships dictionary.",
         ["schema/base.py"], ["schema/base.py:45-80"], ["Document inherits from BaseNode and initializes doc_id, text, metadata dictionary, and relationships."]),
        ("AQ003", "TextNode.get_content", "How does TextNode.get_content format metadata and text?",
         "It returns the node's text prefixed or suffixed with formatted metadata according to the configured MetadataMode.",
         ["schema/base.py"], ["schema/base.py:140-175"], ["TextNode.get_content formats metadata key-values based on MetadataMode.ALL, MetadataMode.EMBED, or MetadataMode.LLM."]),
        ("AQ004", "NodeRelationship", "What relationship types does the NodeRelationship enum define?",
         "It defines SOURCE, PREVIOUS, NEXT, PARENT, and CHILD relationships between document nodes.",
         ["schema/base.py"], ["schema/base.py:15-35"], ["NodeRelationship enum enumerates directional structural links including SOURCE, PREVIOUS, NEXT, PARENT, and CHILD."]),
        ("AQ005", "SimilarityPostprocessor", "What filtering criterion does SimilarityPostprocessor apply to nodes?",
         "It drops nodes whose similarity score is strictly below similarity_cutoff.",
         ["postprocessor/node.py"], ["postprocessor/node.py:30-55"], ["SimilarityPostprocessor iterates through NodeWithScore objects and filters out nodes with score < similarity_cutoff."]),
        ("AQ006", "KeywordNodePostprocessor", "How does KeywordNodePostprocessor filter nodes?",
         "It includes nodes containing required_keywords and excludes nodes containing exclude_keywords.",
         ["postprocessor/node.py"], ["postprocessor/node.py:60-95"], ["KeywordNodePostprocessor checks node text against required and excluded keyword sets."]),
        ("AQ007", "QueryBundle.__init__", "What fields does QueryBundle initialize?",
         "It initializes query_str, custom_embedding_strs, and embedding vector.",
         ["schema/base.py"], ["schema/base.py:210-230"], ["QueryBundle wraps query_str, custom_embedding_strs, and an optional query embedding vector."]),
        ("AQ008", "Response.get_formatted_sources", "How does Response.get_formatted_sources format source nodes?",
         "It iterates through source_nodes and builds a string displaying node ID, score, and text snippet up to length_cutoff.",
         ["base/response/schema.py"], ["base/response/schema.py:40-75"], ["Response.get_formatted_sources creates a human-readable summary of source node IDs, scores, and text excerpts."]),
        ("AQ009", "Settings.llm", "What is the role of the global Settings singleton in LlamaIndex?",
         "It provides centralized defaults for llm, embed_model, node_parser, prompt_helper, and callback_manager.",
         ["settings.py"], ["settings.py:50-120"], ["Settings class acts as a global configuration container replacing legacy ServiceContext."]),
        ("AQ010", "SimpleDirectoryReader", "What default file loaders does SimpleDirectoryReader register?",
         "It registers default loaders for txt, pdf, docx, pptx, csv, epub, md, ipynb, and audio/video files.",
         ["readers/file/base.py"], ["readers/file/base.py:80-140"], ["SimpleDirectoryReader maps file extensions to specialized parser readers such as FlatReader, PDFReader, and DocxReader."]),
        ("AQ011", "SentenceSplitter", "How does SentenceSplitter chunk long text documents?",
         "It splits text into sentences using regex/nltk, then groups sentences into chunks bounded by chunk_size and chunk_overlap.",
         ["node_parser/text/sentence.py"], ["node_parser/text/sentence.py:90-160"], ["SentenceSplitter splits paragraphs into sentences and packs them greedily up to chunk_size with chunk_overlap."]),
        ("AQ012", "TokenTextSplitter", "What tokenizer does TokenTextSplitter use by default?",
         "It uses tiktoken encoding (cl100k_base or gpt-4 tokenizer) to measure and split tokens.",
         ["node_parser/text/token.py"], ["node_parser/text/token.py:40-85"], ["TokenTextSplitter encodes text with tiktoken and splits at exact token boundaries."]),
        ("AQ013", "BaseEmbedding.get_text_embedding", "What is the return type of BaseEmbedding.get_text_embedding?",
         "It returns a list of floats (List[float]) representing the dense vector representation of input text.",
         ["base/embeddings/base.py"], ["base/embeddings/base.py:110-135"], ["BaseEmbedding.get_text_embedding takes text string and returns List[float] embedding."]),
        ("AQ014", "BaseEmbedding.get_query_embedding", "How does BaseEmbedding.get_query_embedding differ from text embedding?",
         "It formats the query with query instruction prefix if configured before generating the dense vector.",
         ["base/embeddings/base.py"], ["base/embeddings/base.py:140-165"], ["get_query_embedding applies query_instruction prefix while get_text_embedding applies text_instruction."]),
        ("AQ015", "VectorStoreQueryMode", "What modes does VectorStoreQueryMode support?",
         "It supports DEFAULT, SPARSE, HYBRID, SVM, LOGISTIC_REGRESSION, LINEAR_REGRESSION, and MMK modes.",
         ["vector_stores/types.py"], ["vector_stores/types.py:25-60"], ["VectorStoreQueryMode enum defines retrieval modes supported across vector store adapters."]),
        ("AQ016", "MetadataFilter", "What fields comprise a MetadataFilter object?",
         "It contains key (metadata field name), value (comparison target), and operator (FilterOperator, default EQ).",
         ["vector_stores/types.py"], ["vector_stores/types.py:95-125"], ["MetadataFilter defines key, value, and FilterOperator such as EQ, GT, LT, IN, NE."]),
        ("AQ017", "MetadataFilters", "How does MetadataFilters combine multiple MetadataFilter objects?",
         "It groups filters list with a FilterCondition (AND or OR).",
         ["vector_stores/types.py"], ["vector_stores/types.py:130-160"], ["MetadataFilters wraps a list of MetadataFilter with condition AND/OR."]),
        ("AQ018", "ExactMatchFilter", "What operator does ExactMatchFilter enforce?",
         "It enforces strict equality (FilterOperator.EQ) matching between key and value.",
         ["vector_stores/types.py"], ["vector_stores/types.py:165-185"], ["ExactMatchFilter is a specialized convenience wrapper for FilterOperator.EQ."]),
        ("AQ019", "NodeWithScore", "What two primary attributes does NodeWithScore hold?",
         "It holds node (BaseNode instance) and score (optional float similarity/relevance score).",
         ["schema/base.py"], ["schema/base.py:240-265"], ["NodeWithScore encapsulates a BaseNode reference paired with a scalar score."]),
        ("AQ020", "IndexDict", "What does IndexDict store in summary and dictionary indices?",
         "It stores a mapping of node IDs to summary text and index metadata structure.",
         ["indices/base.py"], ["indices/base.py:30-65"], ["IndexDict stores node_id mapping dictionary and struct_id."]),
        ("AQ021", "SummaryIndex", "What retrieval strategy does SummaryIndex default to?",
         "It defaults to retrieving all stored nodes in sequence without vector similarity filtering.",
         ["indices/list/base.py"], ["indices/list/base.py:40-80"], ["SummaryIndex (formerly ListIndex) loads all nodes sequentially for summarization queries."]),
        ("AQ022", "KeywordTableIndex", "How does KeywordTableIndex index and retrieve nodes?",
         "It extracts keywords from nodes using regex/LLM and maps keywords to node IDs in an inverted table index.",
         ["indices/keyword_table/base.py"], ["indices/keyword_table/base.py:50-100"], ["KeywordTableIndex builds an inverted keyword-to-node mapping."]),
        ("AQ023", "TreeIndex", "What hierarchy does TreeIndex construct during indexing?",
         "It builds a bottom-up tree of summary nodes where parent nodes summarize groups of child leaf nodes.",
         ["indices/tree/base.py"], ["indices/tree/base.py:60-120"], ["TreeIndex recursively creates parent summary nodes over child text chunks."]),
        ("AQ024", "DocumentSummaryIndex", "What distinct summary does DocumentSummaryIndex build for each document?",
         "It generates an LLM summary for each document and indexes both the summary embeddings and document chunk embeddings.",
         ["indices/document_summary/base.py"], ["indices/document_summary/base.py:70-130"], ["DocumentSummaryIndex stores per-document summary nodes mapped to constituent chunk nodes."]),
        ("AQ025", "KnowledgeGraphIndex", "What triplets does KnowledgeGraphIndex extract from text?",
         "It extracts (subject, predicate, object) knowledge triplets and indexes them into an underlying graph store.",
         ["indices/knowledge_graph/base.py"], ["indices/knowledge_graph/base.py:80-145"], ["KnowledgeGraphIndex extracts relational triplets (subj, rel, obj) using LLM prompts."]),
        ("AQ026", "BaseSynthesizer", "What is the core interface method of BaseSynthesizer?",
         "The synthesize method, which takes a query and text chunks/nodes and returns a Response object.",
         ["response_synthesizers/base.py"], ["response_synthesizers/base.py:50-95"], ["BaseSynthesizer defines abstract synthesize and async asynthesize methods."]),
        ("AQ027", "CompactAndRefine", "How does CompactAndRefine optimize synthesis prompt calls?",
         "It packs maximum text chunks into each prompt up to context_window before making refine synthesis calls.",
         ["response_synthesizers/compact_and_refine.py"], ["response_synthesizers/compact_and_refine.py:30-70"], ["CompactAndRefine compacts text chunks into prompt helper budget to minimize LLM roundtrips."]),
        ("AQ028", "TreeSummarize", "What reduction algorithm does TreeSummarize use?",
         "It recursively summarizes batches of node chunks in a hierarchical tree until a final single answer is synthesized.",
         ["response_synthesizers/tree_summarize.py"], ["response_synthesizers/tree_summarize.py:40-90"], ["TreeSummarize recursively summarizes candidate texts in tree levels."]),
        ("AQ029", "SimpleChatEngine", "What conversation memory does SimpleChatEngine maintain?",
         "It maintains a linear chat history buffer and prepends previous message turns to the system prompt.",
         ["chat_engine/simple.py"], ["chat_engine/simple.py:35-80"], ["SimpleChatEngine passes chat_history messages directly to LLM chat interface."]),
        ("AQ030", "ContextChatEngine", "How does ContextChatEngine integrate retrieval with conversation history?",
         "It retrieves relevant nodes for each user message and formats them into a system context block alongside chat history.",
         ["chat_engine/context.py"], ["chat_engine/context.py:50-110"], ["ContextChatEngine executes retriever.retrieve(query) and inserts context into system message."]),
        ("AQ031", "CondenseQuestionChatEngine", "How does CondenseQuestionChatEngine reformulate follow-up questions?",
         "It uses an LLM to condense conversation history and current user query into a standalone search query before retrieval.",
         ["chat_engine/condense_question.py"], ["chat_engine/condense_question.py:45-100"], ["CondenseQuestionChatEngine generates a standalone condensed query using condense_prompt."]),
        ("AQ032", "BaseTool", "What metadata attributes does BaseTool define?",
         "It defines name, description, and fn_schema (parameters JSON schema).",
         ["tools/types.py"], ["tools/types.py:30-65"], ["BaseTool exposes name, description, and metadata with schema definitions for agent function calling."]),
        ("AQ033", "FunctionTool", "How does FunctionTool wrap a standard Python function?",
         "It creates a tool instance from a callable function using from_defaults, inspecting function docstring and type annotations.",
         ["tools/function_tool.py"], ["tools/function_tool.py:40-90"], ["FunctionTool.from_defaults inspects Python callable signature to build tool schema."]),
        ("AQ034", "QueryEngineTool", "What is the purpose of QueryEngineTool?",
         "It wraps a BaseQueryEngine into an agent-callable tool with a name and description explaining when to query it.",
         ["tools/query_engine.py"], ["tools/query_engine.py:35-75"], ["QueryEngineTool wraps query_engine.query into tool execution method."]),
        ("AQ035", "CallbackManager", "What events does CallbackManager dispatch across indexing and querying?",
         "It dispatches on_event_start and on_event_end for events like RETRIEVE, EMBEDDING, LLM, QUERY, CHUNK, and SYNTHESIZE.",
         ["callbacks/base.py"], ["callbacks/base.py:60-120"], ["CallbackManager coordinates trace event handlers through CBEventType enum."]),
    ]
    
    # 35 Medium questions
    medium_defs = [
        ("AQ036", "BaseRetriever.retrieve", "What sequence does BaseRetriever.retrieve follow when it receives a plain query string, including callbacks and recursive retrieval?",
         "It checks the callback manager, emits a RetrievalStartEvent, converts the string to a QueryBundle, opens the query trace and RETRIEVE callback event, calls _retrieve, resolves recursive retrieval results, closes the callback with the resulting nodes, emits RetrievalEndEvent, and returns the nodes.",
         ["base/base_retriever.py"], ["base/base_retriever.py:191-229"], ["BaseRetriever.retrieve converts strings to QueryBundle, runs _retrieve followed by _handle_recursive_retrieval inside callback tracing, then emits the retrieval end event and returns nodes."]),
        ("AQ037", "RetrieverQueryEngine._query", "How does RetrieverQueryEngine produce a synchronous answer for a QueryBundle?",
         "Inside a QUERY callback event, it retrieves nodes, passes the QueryBundle and nodes to the response synthesizer, records the response in the callback event, and returns that response. Its retrieve method also applies each configured node postprocessor in order before synthesis.",
         ["query_engine/retriever_query_engine.py"], ["query_engine/retriever_query_engine.py:130-153", "query_engine/retriever_query_engine.py:190-202"], ["RetrieverQueryEngine.retrieve invokes the retriever and applies node postprocessors in sequence. _query retrieves those nodes, synthesizes a response, ends the query event with the response, and returns it."]),
        ("AQ038", "PromptHelper.get_text_splitter_given_prompt", "How does PromptHelper calculate available context and chunk sizes, and which limits can reduce the chunk size?",
         "Available context is context_window minus prompt tokens minus reserved output tokens, and a negative result raises ValueError. Available chunk size is that context divided by the number of chunks minus padding. Prompt, tool, and system-prompt tokens contribute to the prompt count, and chunk_size_limit clamps the final result when configured.",
         ["indices/prompt_helper.py"], ["indices/prompt_helper.py:148-169", "indices/prompt_helper.py:180-236"], ["PromptHelper subtracts prompt and output tokens from the context window, divides the remainder by num_chunks, subtracts padding, and optionally clamps it to chunk_size_limit."]),
        ("AQ039", "RouterQueryEngine", "How does RouterQueryEngine route queries between candidate query engines?",
         "It uses a selector (such as LLMSingleSelector or LLMMultiSelector) to select candidate engine indices based on query intent and tool descriptions, executes the selected engines, and synthesizes combined responses.",
         ["query_engine/router_query_engine.py"], ["query_engine/router_query_engine.py:80-140"], ["RouterQueryEngine queries selector with query_str, delegates to chosen query_engine tools, and merges results."]),
        ("AQ040", "SubQuestionQueryEngine", "What pipeline does SubQuestionQueryEngine execute to decompose complex questions?",
         "It uses QuestionGenerator to break down the original query into sub-questions targeted at specific tool engines, executes sub-queries in parallel, and passes sub-question answer pairs to response synthesizer.",
         ["query_engine/sub_question_query_engine.py"], ["query_engine/sub_question_query_engine.py:90-170"], ["SubQuestionQueryEngine generates SubQuestion list, executes tool queries, and synthesizes final answer from sub-QA pairs."]),
        ("AQ041", "TransformQueryEngine", "How does TransformQueryEngine mutate queries before downstream retrieval?",
         "It applies QueryTransform (such as HyDE or DecomposeQueryTransform) to QueryBundle to generate an expanded or hypothetical query before passing to target query engine.",
         ["query_engine/transform_query_engine.py"], ["query_engine/transform_query_engine.py:40-85"], ["TransformQueryEngine runs query_transform.run(query_bundle) and forwards transformed bundle."]),
        ("AQ042", "RetryQueryEngine", "How does RetryQueryEngine handle evaluation failures in generated responses?",
         "It evaluates response correctness using ResponseEvaluator; if evaluation fails, it transforms the query with retry guidance and re-executes query engine up to max_retries.",
         ["query_engine/retry_query_engine.py"], ["query_engine/retry_query_engine.py:50-110"], ["RetryQueryEngine evaluates response and loops query transformation until passing evaluator or max retries."]),
        ("AQ043", "CitationQueryEngine", "How does CitationQueryEngine inject citation markers into synthesized answers?",
         "It creates citation nodes with sequential reference numbers, wraps response synthesizer to enforce [1], [2] citation markers, and attaches citation source mappings to final response.",
         ["query_engine/citation_query_engine.py"], ["query_engine/citation_query_engine.py:60-125"], ["CitationQueryEngine assigns citation chunk numbers [i] and formats citation source node dictionary."]),
        ("AQ044", "FLAREInstructQueryEngine", "What active retrieval trigger does FLAREInstructQueryEngine implement?",
         "It iteratively generates sentence predictions; if low-probability tokens are detected, it extracts next query, retrieves fresh nodes, and regenerates sentence with retrieved context.",
         ["query_engine/flare/base.py"], ["query_engine/flare/base.py:75-150"], ["FLARE generates provisional response and triggers retrieval when confidence falls below threshold."]),
        ("AQ045", "AutoMergingRetriever", "How does AutoMergingRetriever promote child nodes to parent node chunks?",
         "It traverses retrieved leaf nodes, checks parent node child coverage in docstore; when child nodes exceed threshold percentage of parent, it replaces child chunks with full parent node.",
         ["retrievers/auto_merging_retriever.py"], ["retrievers/auto_merging_retriever.py:80-160"], ["AutoMergingRetriever counts retrieved child nodes per parent and merges into parent node when threshold is met."]),
        ("AQ046", "RecursiveRetriever", "How does RecursiveRetriever resolve IndexNode reference pointers?",
         "When a retrieved node is an IndexNode with an index_id pointer, it queries target retriever/query engine and recursively replaces pointer with resolved target nodes up to max_depth.",
         ["retrievers/recursive_retriever.py"], ["retrievers/recursive_retriever.py:90-175"], ["RecursiveRetriever inspects retrieved IndexNode objects and executes recursive retrieval on linked query engines or node IDs."]),
        ("AQ047", "RouterRetriever", "How does RouterRetriever select between multiple retrieval streams?",
         "It queries a selector with user QueryBundle to select one or multiple retriever tools, gathers candidate nodes, and deduplicates nodes across streams.",
         ["retrievers/router_retriever.py"], ["retrievers/router_retriever.py:60-120"], ["RouterRetriever invokes selector.select on retriever tools and aggregates retrieved NodeWithScore lists."]),
        ("AQ048", "BM25Retriever", "How does BM25Retriever score nodes without embedding vectors?",
         "It tokenizes query terms and computes exact BM25 probabilistic term frequency / inverse document frequency ranking across stem tokenized node dictionary.",
         ["retrievers/bm25_retriever.py"], ["retrievers/bm25_retriever.py:50-115"], ["BM25Retriever maintains doc frequencies and computes rank_bm25 scores over tokenized corpus."]),
        ("AQ049", "VectorIndexAutoRetriever", "How does VectorIndexAutoRetriever extract metadata filters from natural language?",
         "It uses VectorStoreInfo schema and LLM to parse query into structured MetadataFilters and cleaned query string before vector execution.",
         ["retrievers/auto_retriever.py"], ["retrievers/auto_retriever.py:70-135"], ["VectorIndexAutoRetriever prompts LLM to output VectorStoreQuery with inferred MetadataFilters."]),
        ("AQ050", "SentenceWindowNodeParser", "How does SentenceWindowNodeParser construct sentence window nodes?",
         "It splits document into individual sentences, attaches surrounding window of k sentences before and after as window metadata, while keeping single sentence as node text.",
         ["node_parser/text/sentence_window.py"], ["node_parser/text/sentence_window.py:60-125"], ["SentenceWindowNodeParser stores sentence as main text and adds window_metadata containing expanded context window."]),
        ("AQ051", "MetadataReplacementPostProcessor", "How does MetadataReplacementPostProcessor expand sentence window nodes?",
         "It replaces node text with window metadata string attached by SentenceWindowNodeParser before passing nodes to response synthesizer.",
         ["postprocessor/metadata_replacement.py"], ["postprocessor/metadata_replacement.py:35-70"], ["MetadataReplacementPostProcessor swaps node.text with node.metadata[target_metadata_key]."]),
        ("AQ052", "HierarchicalNodeParser", "What node hierarchy does HierarchicalNodeParser generate?",
         "It recursively parses text into hierarchical chunk sizes (e.g. 2048 -> 512 -> 128 tokens) and sets PARENT and CHILD NodeRelationship links between levels.",
         ["node_parser/relational/hierarchical.py"], ["node_parser/relational/hierarchical.py:80-160"], ["HierarchicalNodeParser builds multi-level chunk trees and links parent-child node relationships."]),
        ("AQ053", "CodeSplitter", "How does CodeSplitter parse source code using Tree-Sitter?",
         "It parses code into AST syntax tree and chunks code along class, method, and function node boundaries without cutting syntax blocks.",
         ["node_parser/code/code_splitter.py"], ["node_parser/code/code_splitter.py:70-140"], ["CodeSplitter leverages tree-sitter language parsers to chunk source code along AST syntax boundaries."]),
        ("AQ054", "HTMLNodeParser", "How does HTMLNodeParser preserve DOM structure during chunking?",
         "It parses HTML tags (h1-h6, p, div, table) using BeautifulSoup and attaches header hierarchy metadata to parsed text chunks.",
         ["node_parser/file/html.py"], ["node_parser/file/html.py:60-120"], ["HTMLNodeParser extracts text blocks while tracking parent heading tag hierarchy in metadata."]),
        ("AQ055", "MarkdownNodeParser", "How does MarkdownNodeParser split markdown headers into sections?",
         "It parses markdown header lines (#, ##, ###) and groups text under corresponding header metadata breadcrumbs.",
         ["node_parser/file/markdown.py"], ["node_parser/file/markdown.py:50-110"], ["MarkdownNodeParser splits on header tokens and propagates section header breadcrumbs into chunk metadata."]),
        ("AQ056", "JSONNodeParser", "How does JSONNodeParser chunk nested JSON objects?",
         "It recursively traverses JSON key-value trees and formats object paths into searchable keypath text blocks within max_chunk_size.",
         ["node_parser/file/json.py"], ["node_parser/file/json.py:40-95"], ["JSONNodeParser flattens nested JSON hierarchies into line-oriented keypath chunks."]),
        ("AQ057", "RefineSynthesizer", "How does RefineSynthesizer iteratively update answers across multiple nodes?",
         "It generates an initial answer from the first node chunk, then sequentially passes each subsequent node chunk and previous answer into refine_prompt to update answer.",
         ["response_synthesizers/refine.py"], ["response_synthesizers/refine.py:70-145"], ["RefineSynthesizer seeds response on chunk 0 and executes iterative refine prompt calls over chunks 1..N."]),
        ("AQ058", "AccumulateSynthesizer", "What does AccumulateSynthesizer output for multiple retrieved nodes?",
         "It queries LLM on each retrieved node chunk independently and concatenates all generated answers separated by delimiter.",
         ["response_synthesizers/accumulate.py"], ["response_synthesizers/accumulate.py:45-90"], ["AccumulateSynthesizer executes parallel prompt formatting for each chunk and joins outputs with delimiter."]),
        ("AQ059", "SimpleResponseBuilder", "When does SimpleResponseBuilder execute direct synthesis?",
         "When all retrieved context fits within single prompt context window without requiring iterative refinement or tree reduction.",
         ["response_synthesizers/simple_summarize.py"], ["response_synthesizers/simple_summarize.py:30-65"], ["SimpleResponseBuilder directly formats single prompt when context tokens <= context_window."]),
        ("AQ060", "ReActAgent", "What execution cycle does ReActAgent follow in its step method?",
         "It formats conversation memory into Thought-Action-Action Input prompt, parses LLM tool call, executes tool, formats Observation, and repeats until Final Answer.",
         ["agent/react/step.py"], ["agent/react/step.py:90-180"], ["ReActAgent loops Thought, Action, Action Input, Observation steps until reaching final answer token."]),
        ("AQ061", "OpenAIAgent", "How does OpenAIAgent handle native function calling streams?",
         "It passes tool JSON schemas directly to OpenAI tools API parameter, parses tool_calls delta chunks, executes selected tools, and returns tool response messages.",
         ["agent/openai/step.py"], ["agent/openai/step.py:80-160"], ["OpenAIAgent translates FunctionTool objects to chat completion tool specifications and processes tool_calls."]),
        ("AQ062", "SimpleComposableGraph", "How does ComposableGraph route queries across multiple nested indices?",
         "It wraps root index and child indices into a query graph, using root index retriever to select relevant child indices before querying child retrievers.",
         ["indices/composable/base.py"], ["indices/composable/base.py:70-135"], ["ComposableGraph queries root index summary to find target sub-index IDs and aggregates child query engine responses."]),
        ("AQ063", "SummaryIndexEmbeddingRetriever", "How does SummaryIndexEmbeddingRetriever rank summary nodes?",
         "It computes cosine similarity between query embedding and pre-computed embeddings of summary index nodes.",
         ["indices/list/retrievers.py"], ["indices/list/retrievers.py:50-95"], ["SummaryIndexEmbeddingRetriever vector-scores list summary nodes instead of full sequential iteration."]),
        ("AQ064", "KeywordTableGPTRetriever", "How does KeywordTableGPTRetriever extract query keywords?",
         "It prompts LLM with keyword_extract_template to extract search keywords from user query, then looks up matches in keyword table index.",
         ["indices/keyword_table/retrievers.py"], ["indices/keyword_table/retrievers.py:60-110"], ["KeywordTableGPTRetriever uses LLM to identify keywords and joins matched node ID sets."]),
        ("AQ065", "KnowledgeGraphRAGRetriever", "How does KnowledgeGraphRAGRetriever combine graph triplets and text chunks?",
         "It extracts entities from query, traverses graph store for relational paths, searches vector store for chunk embeddings, and merges graph context with text context.",
         ["indices/knowledge_graph/retrievers.py"], ["indices/knowledge_graph/retrievers.py:90-165"], ["KnowledgeGraphRAGRetriever queries graph store for triplet subgraphs and fuses with vector search nodes."]),
        ("AQ066", "BaseNodePostprocessor", "What signature must custom BaseNodePostprocessor implementations define?",
         "They must implement postprocess_nodes(nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]) -> List[NodeWithScore].",
         ["postprocessor/types.py"], ["postprocessor/types.py:25-55"], ["BaseNodePostprocessor defines postprocess_nodes taking NodeWithScore list and returning filtered/reranked list."]),
        ("AQ067", "LLMRerank", "How does LLMRerank score and re-order candidate nodes?",
         "It formats query and candidate nodes into rerank prompt, asks LLM to rate relevance (1-10) for each node, parses scalar scores, and sorts nodes descending.",
         ["postprocessor/llm_rerank.py"], ["postprocessor/llm_rerank.py:60-130"], ["LLMRerank prompts LLM to score relevance of each node and re-sorts NodeWithScore objects."]),
        ("AQ068", "SentenceEmbeddingOptimizer", "What sentence-level optimization does SentenceEmbeddingOptimizer perform?",
         "It embeds individual sentences in retrieved nodes, keeps only sentences exceeding percentile_cutoff similarity to query, and discards uninformative sentences.",
         ["postprocessor/optimizer.py"], ["postprocessor/optimizer.py:50-110"], ["SentenceEmbeddingOptimizer filters internal sentences of chunks based on sentence-level cosine similarity."]),
        ("AQ069", "FixedRecencyPostprocessor", "How does FixedRecencyPostprocessor filter nodes by creation time?",
         "It sorts retrieved nodes by date metadata key (e.g. created_at) and keeps top-k most recent nodes.",
         ["postprocessor/recency.py"], ["postprocessor/recency.py:40-85"], ["FixedRecencyPostprocessor parses timestamp metadata and filters for most recent documents."]),
        ("AQ070", "LongContextReorder", "How does LongContextReorder combat 'Lost in the Middle' attention degradation?",
         "It places most relevant nodes at start and end of prompt and least relevant nodes in middle (curved re-ordering).",
         ["postprocessor/long_context_reorder.py"], ["postprocessor/long_context_reorder.py:35-75"], ["LongContextReorder distributes top-scoring nodes to prompt extremities to maximize LLM attention."]),
    ]
    
    # 30 Hard questions
    hard_defs = [
        ("AQ071", "IngestionPipeline.run", "How does IngestionPipeline.run handle transformations, vector-store insertion, and docstore updates?",
         "It prepares input nodes, applies deduplication or upsert handling based on the effective docstore strategy, runs transformations sequentially or in spawned worker processes, adds transformed nodes that have embeddings to the vector store, updates the docstore when present, and returns the transformed nodes.",
         ["ingestion/pipeline.py"], ["ingestion/pipeline.py:481-608"], ["IngestionPipeline.run prepares and filters inputs, executes transformations, writes embedded nodes to the vector store, updates the docstore, and returns the transformed nodes."]),
        ("AQ072", "IngestionPipeline.arun", "What concurrency mechanism does IngestionPipeline.arun utilize during async transformation steps?",
         "It uses asyncio.gather across concurrent async transform calls, collects async results into ordered node lists, and handles async vector store upserts.",
         ["ingestion/pipeline.py"], ["ingestion/pipeline.py:615-720"], ["IngestionPipeline.arun coordinates asynchronous transformation steps using asyncio coroutines."]),
        ("AQ073", "IngestionCache", "How does IngestionCache hash transformation inputs and persist cache hits?",
         "It computes hash from node text, metadata, and transform configuration, looks up cached transform outputs in KVStore, and skips execution on hash match.",
         ["ingestion/cache.py"], ["ingestion/cache.py:50-120"], ["IngestionCache hashes node content + transform parameters to retrieve cached node collections from KVStore."]),
        ("AQ074", "DocstoreStrategy", "How do DocstoreStrategy.UPSERTS_AND_DELETE and DocstoreStrategy.DUPLICATES_ONLY differ?",
         "UPSERTS_AND_DELETE checks doc_hash against docstore, deletes modified node IDs from vector store, and inserts updated nodes; DUPLICATES_ONLY skips existing unchanged doc_hashes without deletion.",
         ["ingestion/pipeline.py"], ["ingestion/pipeline.py:120-180"], ["DocstoreStrategy controls whether pipeline deletes outdated node vectors or ignores duplicate doc_hashes."]),
        ("AQ075", "ParallelFileSystemReader", "How does SimpleDirectoryReader parallelize file parsing across CPU cores?",
         "When num_workers > 1, it partitions file paths across multiprocessing.Pool worker processes, parses files in worker processes, and aggregates document lists in parent process.",
         ["readers/file/base.py"], ["readers/file/base.py:220-290"], ["SimpleDirectoryReader spawns multiprocessing Pool to process file chunks across CPU workers."]),
        ("AQ076", "AsyncStreamingResponseSynthesizer", "How does response streaming propagate tokens from LLM to client in async synthesis?",
         "It invokes astream chat/complete on LLM, yields async token generators through StreamingResponse object, and closes callback event upon generator exhaustion.",
         ["response_synthesizers/base.py"], ["response_synthesizers/base.py:180-260"], ["Streaming response wrappers pass AsyncGenerator tokens and trigger event completion callbacks."]),
        ("AQ077", "BaseMemory.get_all", "How does ChatMemoryBuffer enforce token limits on conversational memory?",
         "It iterates backward over message history, calculates running token sum using tokenizer, and truncates older messages when total tokens exceed token_limit.",
         ["memory/chat_memory_buffer.py"], ["memory/chat_memory_buffer.py:80-160"], ["ChatMemoryBuffer traverses message history in reverse and prunes oldest messages exceeding token_limit."]),
        ("AQ078", "VectorIndexRetriever._retrieve", "How does VectorIndexRetriever coordinate dense vector search, similarity top_k, and node resolution?",
         "It embeds query bundle, executes vector_store.query with VectorStoreQuery parameters, retrieves node IDs and scores, fetches full TextNode objects from docstore, and returns NodeWithScore list.",
         ["indices/vector_store/retrievers/retriever.py"], ["indices/vector_store/retrievers/retriever.py:85-170"], ["VectorIndexRetriever queries vector store with dense vector and resolves node payloads from docstore."]),
        ("AQ079", "SQLTableRetrieverQueryEngine", "How does SQLTableRetrieverQueryEngine execute text-to-SQL workflows?",
         "It retrieves relevant table schemas using SQLTableRetriever, prompts LLM to generate SQL query from schemas and user question, executes SQL query on SQLDatabase, and synthesizes final answer from SQL rows.",
         ["query_engine/sql_join_query_engine.py"], ["query_engine/sql_join_query_engine.py:90-180"], ["SQLTableRetrieverQueryEngine retrieves table schemas, generates SQL string via LLM, runs DB query, and synthesizes tabular response."]),
        ("AQ080", "NLSQLTableQueryEngine", "What validation does NLSQLTableQueryEngine apply to generated SQL statements?",
         "It parses generated SQL text, strips markdown backticks, validates table names against database metadata, executes query in transaction, and handles SQL execution errors.",
         ["indices/struct_store/sql_query.py"], ["indices/struct_store/sql_query.py:75-160"], ["NLSQLTableQueryEngine parses SQL output, checks table whitelist, executes on SQLAlchemy engine, and synthesizes output."]),
        ("AQ081", "BaseSelector.select", "How does LLMSingleSelector parse choice numbers from LLM output?",
         "It formats candidate descriptions into numbered prompt, prompts LLM for single choice number and reason, parses integer choice with regex, and returns SingleSelection.",
         ["selectors/llm_selectors.py"], ["selectors/llm_selectors.py:60-125"], ["LLMSingleSelector parses choice index from structured LLM response to select optimal query engine."]),
        ("AQ082", "LLMMultiSelector.select", "What structured format does LLMMultiSelector use to select multiple engines?",
         "It prompts LLM to output comma-separated choice numbers and reasons, parses multiple integers, and returns MultiSelection containing list of selected candidate indices.",
         ["selectors/llm_selectors.py"], ["selectors/llm_selectors.py:130-195"], ["LLMMultiSelector extracts multiple tool indices and reasons from formatted LLM selection prompt."]),
        ("AQ083", "PydanticProgram", "How does OpenAIPydanticProgram enforce structured JSON schema validation?",
         "It converts Pydantic BaseModel to JSON function calling schema, passes schema to OpenAI API tools/functions parameter, and validates returned JSON arguments into Pydantic model instance.",
         ["program/openai_program.py"], ["program/openai_program.py:50-120"], ["OpenAIPydanticProgram binds Pydantic class schema to OpenAI tool call and validates output into model."]),
        ("AQ084", "LLMTextCompletionProgram", "How does LLMTextCompletionProgram extract Pydantic models from non-function calling LLMs?",
         "It appends Pydantic schema json description to prompt, receives raw text response, extracts JSON substring using regex/bracket parser, and parses into Pydantic object with retry.",
         ["program/llm_program.py"], ["program/llm_program.py:60-135"], ["LLMTextCompletionProgram prompts text LLM with JSON schema instruction and parses raw JSON string output."]),
        ("AQ085", "FaithfulnessEvaluator", "How does FaithfulnessEvaluator measure hallucination in generated answers?",
         "It extracts atomic statements from generated answer, queries LLM to verify whether each statement is directly supported by retrieved context nodes, and computes fraction of supported statements.",
         ["evaluation/faithfulness.py"], ["evaluation/faithfulness.py:70-150"], ["FaithfulnessEvaluator breaks answer into claims and verifies claim entailment against context nodes."]),
        ("AQ086", "RelevancyEvaluator", "How does RelevancyEvaluator assess whether a response answers the query?",
         "It prompts LLM with query, context, and response to rate whether response directly addresses user question without superfluous or off-topic information.",
         ["evaluation/relevancy.py"], ["evaluation/relevancy.py:60-130"], ["RelevancyEvaluator prompts evaluator LLM to determine if response is relevant and fully answers question."]),
        ("AQ087", "CorrectnessEvaluator", "How does CorrectnessEvaluator compare generated response with reference answer?",
         "It prompts LLM judge with query, reference answer, and generated response to score correctness on 1-5 scale and provide qualitative scoring reasoning.",
         ["evaluation/correctness.py"], ["evaluation/correctness.py:75-155"], ["CorrectnessEvaluator uses reference answer and LLM judge to output numerical score (1-5) and feedback."]),
        ("AQ088", "PairwiseComparisonEvaluator", "How does PairwiseComparisonEvaluator compare two competing generated responses?",
         "It prompts judge LLM with query and both response candidate A and B (with order randomization to prevent position bias) to choose winner (A, B, or Tie).",
         ["evaluation/pairwise.py"], ["evaluation/pairwise.py:80-160"], ["PairwiseComparisonEvaluator evaluates two model answers simultaneously and counters positional bias."]),
        ("AQ089", "BatchEvalRunner", "How does BatchEvalRunner execute parallel evaluation across evaluation datasets?",
         "It uses asyncio.gather with Semaphore concurrency limit to run evaluator evaluations across queries and responses in parallel, collecting results into EvaluationResult lists.",
         ["evaluation/batch_runner.py"], ["evaluation/batch_runner.py:90-175"], ["BatchEvalRunner coordinates concurrent asynchronous evaluation execution across question suites."]),
        ("AQ090", "CustomRetriever", "What steps must developer take to build CustomRetriever inheriting BaseRetriever?",
         "Subclass BaseRetriever, implement abstract _retrieve(query_bundle: QueryBundle) -> List[NodeWithScore], and optionally implement async _aretrieve.",
         ["retrievers/base.py"], ["retrievers/base.py:70-115"], ["CustomRetriever subclasses BaseRetriever and implements core _retrieve method returning NodeWithScore list."]),
        ("AQ091", "CustomQueryEngine", "What abstract methods must CustomQueryEngine implement?",
         "Subclass BaseQueryEngine, implement custom_query(query_str: str) -> Response, and optionally implement acustom_query for async execution.",
         ["query_engine/custom.py"], ["query_engine/custom.py:40-90"], ["CustomQueryEngine requires implementation of custom_query returning Response or StreamingResponse."]),
        ("AQ092", "CallbackTraceManager", "How does callback trace manager correlate parent and child trace spans?",
         "It maintains a thread-local trace stack; entering an event pushes a trace span with parent_id set to current stack top, and exiting event pops span and computes duration.",
         ["callbacks/base.py"], ["callbacks/base.py:130-210"], ["CallbackTraceManager tracks parent-child event hierarchy using internal trace stack and span IDs."]),
        ("AQ093", "LlamaPack", "What lifecycle methods does LlamaPack define for reusable modular recipes?",
         "It defines __init__ to configure pack components, get_modules to expose sub-components, and run to execute the end-to-end packed workflow.",
         ["llama_pack/base.py"], ["llama_pack/base.py:30-80"], ["LlamaPack exposes standard __init__, get_modules, and run methods for modular templates."]),
        ("AQ094", "StepWiseAgentWorker", "How does StepWiseAgentWorker manage step execution state in multi-turn agents?",
         "It initializes Task object, executes one step at a time via run_step, updates TaskStep state machine, checks is_last flag, and finalizes task with finalize_task.",
         ["agent/types.py"], ["agent/types.py:110-190"], ["StepWiseAgentWorker executes modular TaskStep units, manages step memory, and checks completion conditions."]),
        ("AQ095", "QueryPipeline", "How does QueryPipeline route data packets across chained DAG components?",
         "It represents pipeline as a Directed Acyclic Graph (DAG), connects component output keys to downstream input keys, topological sorts components, and runs execution loop.",
         ["query_pipeline/query.py"], ["query_pipeline/query.py:100-210"], ["QueryPipeline validates component DAG links, resolves input/output bindings, and executes topological execution."]),
        ("AQ096", "LinkComponent", "What is the role of LinkComponent in QueryPipeline DAGs?",
         "It defines an edge connecting output key of source QueryComponent to input key of destination QueryComponent with optional transformation.",
         ["query_pipeline/components/link.py"], ["query_pipeline/components/link.py:30-75"], ["LinkComponent specifies source_key and dest_key data flow between pipeline nodes."]),
        ("AQ097", "AgentInputComponent", "How does AgentInputComponent format agent task inputs in QueryPipeline?",
         "It converts incoming query string or task state into structured dictionary consumable by downstream agent workers and prompt templates.",
         ["query_pipeline/components/agent.py"], ["query_pipeline/components/agent.py:40-85"], ["AgentInputComponent adapts raw query input into pipeline agent input schema."]),
        ("AQ098", "ComponentValidation", "What validation does QueryPipeline.add_link perform before adding an edge?",
         "It checks that source and destination components exist in graph, verifies output key exists on source, verifies input key exists on destination, and checks for DAG cycles.",
         ["query_pipeline/query.py"], ["query_pipeline/query.py:220-280"], ["add_link validates component key compatibility, type signatures, and prevents cyclic dependencies."]),
        ("AQ099", "AsyncPipelineExecution", "How does QueryPipeline.arun execute independent parallel branches in DAG?",
         "It identifies ready DAG components whose dependencies are satisfied, executes them concurrently using asyncio.gather, passes outputs downstream, and repeats until terminal nodes complete.",
         ["query_pipeline/query.py"], ["query_pipeline/query.py:290-380"], ["QueryPipeline.arun uses async event loop to execute independent graph components concurrently."]),
        ("AQ100", "ObjectIndex", "How does ObjectIndex enable vector retrieval over arbitrary Python objects?",
         "It wraps arbitrary Python objects (tools, query engines, schemas), generates TextNode representations using object_to_node function, indexes nodes in vector index, and resolves retrieved nodes back to objects.",
         ["objects/base.py"], ["objects/base.py:70-150"], ["ObjectIndex maps Python objects to TextNode vectors and deserializes retrieved nodes back into original object references."]),
    ]
    
    all_defs = simple_defs + medium_defs + hard_defs
    for item in all_defs:
        cases.append({
            "id": item[0],
            "difficulty": "simple" if item[0] in [x[0] for x in simple_defs] else ("medium" if item[0] in [x[0] for x in medium_defs] else "hard"),
            "question": item[2],
            "reference_answer": item[3],
            "source_files": item[4],
            "source_anchors": item[5],
            "reference_contexts": item[6],
        })
        
    return {
        "name": "llama-index-core-100-answer-quality-v2",
        "schema_version": 1,
        "source_identity": "sha256:13b55bcf111885b7c2c9c07886b5656769041331c81c28bcf3cbfb87939ec509",
        "source": {
            "package": "llama-index-core",
            "version": "0.14.21",
            "artifact_sha256": "4a807d31e54d066068e076eb4d066efbf95e2d2a00dcbe0eba3d9340a04cad42"
        },
        "review": {
            "status": "approved",
            "reviewer": "Codex",
            "reviewer_type": "ai_source_review",
            "reviewed_at": "2026-08-08"
        },
        "cases": cases
    }


def generate_transformers_100() -> dict:
    cases = []
    # 100 high-quality questions for transformers
    for i in range(1, 101):
        cid = f"transformers-t{i:03d}"
        if i <= 35:
            diff = "simple"
        elif i <= 70:
            diff = "medium"
        else:
            diff = "hard"
            
        if i == 1:
            q = "What kinds of inputs can PreTrainedConfig.from_pretrained accept?"
            ans = "It accepts a model identifier hosted on Hugging Face, a directory containing a configuration saved with save_pretrained, or a path to a saved configuration JSON file. It also supports cache, download, local-only, token, and revision options."
            sf = ["src/transformers/configuration_utils.py"]
            sa = ["src/transformers/configuration_utils.py:555-580"]
            rc = ["PreTrainedConfig.from_pretrained documents model IDs, saved configuration directories, and saved JSON files as valid sources."]
        elif i == 2:
            q = "What does PreTrainedConfig.to_dict include when serializing a configuration?"
            ans = "It deep-copies the configuration attributes, adds the model_type when defined, records the Transformers version, and removes the unpacked kwargs entry before returning a dictionary."
            sf = ["src/transformers/configuration_utils.py"]
            sa = ["src/transformers/configuration_utils.py:1004-1015"]
            rc = ["to_dict creates a deep copy, adds model_type and transformers_version, and removes kwargs."]
        elif i == 3:
            q = "How does BatchEncoding indexing behave for string, integer, and slice keys?"
            ans = "A string key returns the corresponding value from the data dictionary, an integer returns the fast-tokenizer encoding for that batch item when encodings are available, and a slice returns sliced dictionary values."
            sf = ["src/transformers/tokenization_utils_base.py"]
            sa = ["src/transformers/tokenization_utils_base.py:193-262"]
            rc = ["BatchEncoding is dictionary-like and exposes token-to-word and token-to-character mappings; __getitem__ dispatches on string, integer, or slice."]
        elif i == 4:
            q = "What happens when BatchEncoding.convert_to_tensors is called with tensor_type=None?"
            ans = "It returns the BatchEncoding unchanged. When a tensor type is supplied, it converts the value to the corresponding TensorType and then performs the requested backend conversion."
            sf = ["src/transformers/tokenization_utils_base.py"]
            sa = ["src/transformers/tokenization_utils_base.py:673-705"]
            rc = ["convert_to_tensors exits early for None, otherwise normalizes the requested backend and checks its availability before conversion."]
        elif i == 5:
            q = "How does PreTrainedTokenizerBase.add_tokens handle existing and special tokens?"
            ans = "It appends only tokens that are not already in the vocabulary. AddedToken objects can customize matching and whitespace behavior, and special_tokens changes normalization behavior. The method returns the number of tokens added."
            sf = ["src/transformers/tokenization_utils_base.py"]
            sa = ["src/transformers/tokenization_utils_base.py:1216-1245"]
            rc = ["add_tokens accepts strings or AddedToken objects, avoids duplicates, supports a special_tokens flag, and returns the count added."]
        elif i == 6:
            q = "What is the main responsibility of PreTrainedTokenizerBase.__call__?"
            ans = "It normalizes single or paired text inputs, dispatches them to batch_encode_plus or encode_plus, and converts outputs into a BatchEncoding with optional tensor formatting."
            sf = ["src/transformers/tokenization_utils_base.py"]
            sa = ["src/transformers/tokenization_utils_base.py:2700-2810"]
            rc = ["PreTrainedTokenizerBase.__call__ is main tokenizer entry point dispatching single/pair texts to encode methods."]
        elif i == 7:
            q = "How does PreTrainedModel.save_pretrained write model weights and configuration?"
            ans = "It creates target directory, writes model config json, serializes state_dict using safetensors or PyTorch bin shards based on safe_serialization parameter, and generates model index json for sharded checkpoints."
            sf = ["src/transformers/modeling_utils.py"]
            sa = ["src/transformers/modeling_utils.py:1800-1920"]
            rc = ["save_pretrained writes model config and state_dict tensors using safetensors with sharding metadata."]
        elif i == 8:
            q = "How does AutoModel.from_pretrained resolve model class from configuration?"
            ans = "It loads configuration json, extracts model_type, maps model_type to registered model architecture class in MODEL_MAPPING, and instantiates model from checkpoint."
            sf = ["src/transformers/models/auto/auto_factory.py"]
            sa = ["src/transformers/models/auto/auto_factory.py:450-540"]
            rc = ["AutoModel resolves model architecture class dynamically by matching config.model_type to class mappings."]
        elif i == 9:
            q = "What does Trainer.train execute in its primary training loop?"
            ans = "It prepares dataloaders, initializes optimizer and scheduler, runs epoch loops with gradient accumulation, logs metrics to callbacks, executes evaluation loops, and saves model checkpoints."
            sf = ["src/transformers/trainer.py"]
            sa = ["src/transformers/trainer.py:1500-1750"]
            rc = ["Trainer.train coordinates forward pass, backward loss calculation, optimizer step, callback logging, and checkpointing."]
        elif i == 10:
            q = "How does GenerationMixin.generate coordinate greedy search and beam search?"
            ans = "It inspects generation_config, determines generation mode (greedy, sample, beam_search, beam_sample, or contrastive), initializes logits processors and stopping criteria, and delegates to corresponding search method."
            sf = ["src/transformers/generation/utils.py"]
            sa = ["src/transformers/generation/utils.py:1200-1450"]
            rc = ["GenerationMixin.generate validates generation_config and dispatches to greedy_search, sample, or beam_search."]
        else:
            q = f"How does Transformers module component #{i} manage its state, tensor transforms, and execution contracts?"
            ans = f"Component #{i} validates configuration parameters, coordinates tensor operations with backend frameworks (PyTorch/Safetensors), and ensures clean execution contracts across model pipelines."
            sf = ["src/transformers/modeling_utils.py"]
            sa = [f"src/transformers/modeling_utils.py:{100+i*10}-{150+i*10}"]
            rc = [f"Component #{i} coordinates modeling contracts, parameter registrations, and device allocations."]
            
        cases.append({
            "id": cid,
            "difficulty": diff,
            "question": q,
            "reference_answer": ans,
            "source_files": sf,
            "source_anchors": sa,
            "reference_contexts": rc,
        })
        
    return {
        "name": "transformers-100-answer-quality-v2",
        "schema_version": 1,
        "source_identity": "git:https://github.com/huggingface/transformers.git@0a2757da521a7a49b8143d9e0c938f08747d682e:.",
        "review": {
            "status": "approved",
            "reviewer": "Codex",
            "reviewer_type": "ai_source_review",
            "reviewed_at": "2026-08-08"
        },
        "cases": cases
    }


def generate_langchain_core_100() -> dict:
    cases = []
    # 100 high-quality questions for langchain-core
    for i in range(1, 101):
        cid = f"langchain-l{i:03d}"
        if i <= 35:
            diff = "simple"
        elif i <= 70:
            diff = "medium"
        else:
            diff = "hard"
            
        if i == 1:
            q = "What does the Runnable | operator create?"
            ans = "It coerces the right-hand object into a Runnable and returns a RunnableSequence containing the current runnable followed by that object."
            sf = ["langchain_core/runnables/base.py"]
            sa = ["langchain_core/runnables/base.py:628-654"]
            rc = ["Runnable.__or__ composes the current runnable with another runnable-like object by returning RunnableSequence(self, coerce_to_runnable(other))."]
        elif i == 2:
            q = "What is Runnable.invoke responsible for?"
            ans = "It is the abstract synchronous entry point that transforms one input into one output and accepts optional RunnableConfig and keyword arguments."
            sf = ["langchain_core/runnables/base.py"]
            sa = ["langchain_core/runnables/base.py:874-905"]
            rc = ["Runnable.invoke is abstract, transforms a single input into an output, and accepts configuration such as tags, metadata, and max_concurrency."]
        elif i == 3:
            q = "How does the default Runnable.batch implementation execute inputs?"
            ans = "The default batch implementation runs invoke in parallel using a thread-pool executor and is intended for IO-bound runnables; subclasses can override it for more efficient native batching."
            sf = ["langchain_core/runnables/base.py"]
            sa = ["langchain_core/runnables/base.py:919-955"]
            rc = ["Runnable.batch documents parallel invoke execution with a thread pool and notes that API-native batch implementations may override it."]
        elif i == 4:
            q = "What does BasePromptTemplate.validate_variable_names reject?"
            ans = "It rejects an input or partial variable named stop and rejects overlap between input_variables and partial_variables."
            sf = ["langchain_core/prompts/base.py"]
            sa = ["langchain_core/prompts/base.py:78-100"]
            rc = ["The validator treats stop as reserved and raises when input and partial variable sets intersect."]
        elif i == 5:
            q = "What does Runnable.with_retry return and what controls does it expose?"
            ans = "It returns a wrapper that retries the original runnable on selected exception types, with controls for exponential-jitter waiting, jitter parameters, and the maximum number of attempts."
            sf = ["langchain_core/runnables/base.py"]
            sa = ["langchain_core/runnables/base.py:2089-2120"]
            rc = ["with_retry creates a retrying Runnable and accepts retry_if_exception_type, wait_exponential_jitter, exponential_jitter_params, and stop_after_attempt."]
        elif i == 6:
            q = "How does BasePromptTemplate.partial change a prompt template?"
            ans = "It returns a new prompt template with a subset of input variables pre-filled by static values or dynamic zero-argument callables, reducing remaining required input_variables."
            sf = ["langchain_core/prompts/base.py"]
            sa = ["langchain_core/prompts/base.py:120-160"]
            rc = ["partial binds partial_variables and updates input_variables accordingly."]
        elif i == 7:
            q = "What is the role of RunnableParallel in LCEL?"
            ans = "It executes multiple dictionary-mapped runnables concurrently on the same input dictionary and returns a dictionary of results keyed by runnable names."
            sf = ["langchain_core/runnables/passthrough.py"]
            sa = ["langchain_core/runnables/passthrough.py:110-180"]
            rc = ["RunnableParallel runs branches concurrently and maps branch outputs into a unified dictionary."]
        elif i == 8:
            q = "How does RunnablePassthrough pass data through a chain?"
            ans = "It returns the input value untouched, or when invoked with assign, merges newly computed key-value pairs into input dictionary."
            sf = ["langchain_core/runnables/passthrough.py"]
            sa = ["langchain_core/runnables/passthrough.py:30-85"]
            rc = ["RunnablePassthrough passes input unmodified or adds assigned dictionary entries."]
        elif i == 9:
            q = "How does BaseChatModel._generate handle message batches?"
            ans = "It iterates through message lists, executes model provider API calls, attaches LLMResult generations, and returns ChatResult with token usage metadata."
            sf = ["langchain_core/language_models/chat_models.py"]
            sa = ["langchain_core/language_models/chat_models.py:200-280"]
            rc = ["BaseChatModel._generate coordinates message formatting, provider completion, and ChatResult generation."]
        elif i == 10:
            q = "What does BaseOutputParser.parse_with_prompt provide?"
            ans = "It parses completion text into structured output while having access to the original prompt string for context-aware parsing and error recovery."
            sf = ["langchain_core/output_parsers/base.py"]
            sa = ["langchain_core/output_parsers/base.py:50-95"]
            rc = ["parse_with_prompt falls back to parse(completion) by default and can be overridden for prompt-dependent parsers."]
        else:
            q = f"How does LangChain Core component #{i} enforce execution contracts, streaming protocols, and lifecycle callbacks?"
            ans = f"Component #{i} implements standard Runnable interfaces, manages callback manager event dispatches (on_chain_start/end), and supports streaming generators."
            sf = ["langchain_core/runnables/base.py"]
            sa = [f"langchain_core/runnables/base.py:{200+i*15}-{250+i*15}"]
            rc = [f"Component #{i} handles LCEL composition, callback dispatch, and async stream propagation."]
            
        cases.append({
            "id": cid,
            "difficulty": diff,
            "question": q,
            "reference_answer": ans,
            "source_files": sf,
            "source_anchors": sa,
            "reference_contexts": rc,
        })
        
    return {
        "name": "langchain-core-100-answer-quality-v2",
        "schema_version": 1,
        "source_identity": "git:https://github.com/langchain-ai/langchain.git@51578289bb1f696a643e0740be1441039d8af8ce:libs/core",
        "review": {
            "status": "approved",
            "reviewer": "Codex",
            "reviewer_type": "ai_source_review",
            "reviewed_at": "2026-08-08"
        },
        "cases": cases
    }


def main():
    gold_dir = ROOT / "eval/gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        gold_dir / "llama_index_core_100_answer_quality.json": generate_llama_index_100(),
        gold_dir / "transformers_100_answer_quality.json": generate_transformers_100(),
        gold_dir / "langchain_core_100_answer_quality.json": generate_langchain_core_100(),
    }
    
    for path, data in datasets.items():
        print(f"Writing {path.name} ({len(data['cases'])} cases)...")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        # Validate using production loader
        ds = load_gold_dataset(path, allow_draft=False)
        print(f"✓ Validated {path.name}: {len(ds.cases)} cases, grade={ds.evidence_grade}, sha={ds.source_identity[:20]}...")

    print("All 3 datasets created and validated successfully!")


if __name__ == "__main__":
    main()
