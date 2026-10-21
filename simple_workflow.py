"""
Simplified LangGraph workflow for the matching agent.
This provides a minimal working implementation.
"""

from typing import Dict, Any, List, Optional
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_state import MatchingAgentState, update_workflow_phase
from agent_tools import ALL_TOOLS, get_tools_by_category

logger = logging.getLogger(__name__)

class SimpleMatchingWorkflow:
    """Simplified LangGraph workflow for job-resume matching."""
    
    def __init__(self, llm: ChatOpenAI, checkpointer: Optional[SqliteSaver] = None):
        """Initialize the workflow."""
        self.llm = llm
        self.checkpointer = checkpointer or SqliteSaver.from_conn_string(":memory:")
        self.graph = self._build_graph()
        logger.info("Simple workflow initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build a simple linear workflow."""
        builder = StateGraph(MatchingAgentState)
        
        # Add nodes
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", ToolNode(ALL_TOOLS))
        
        # Simple flow: agent -> tools -> agent (loop until done)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", "__end__": "__end__"}
        )
        builder.add_edge("tools", "agent")
        
        # Compile without checkpointer for now to avoid config issues
        return builder.compile()
    
    def _agent_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Main agent node that processes user queries."""
        try:
            messages = state.get("messages", [])
            
            # Check if this is a search request and try to detect intent
            last_message = messages[-1] if messages else None
            user_query = last_message.content.lower() if last_message else ""
            
            # Handle contact information requests
            if any(keyword in user_query for keyword in ["contact", "email", "phone", "details"]):
                contact_info = self._get_contact_information(user_query)
                response = AIMessage(content=contact_info)
                return {
                    "messages": messages + [response],
                    "workflow_phase": "contact_provided"
                }
            
            # Direct tool usage for common search queries
            if any(keyword in user_query for keyword in ["find", "search", "candidates", "resume", "java", "python", "developer"]):
                # Create a helpful response that includes actual search results
                search_keywords = self._extract_search_keywords(user_query)
                
                if search_keywords:
                    # Perform actual search using tools
                    search_results = self._perform_search(search_keywords)
                    response_content = f"""I'll search for candidates with {search_keywords} experience. Here are the results:

{search_results}

These candidates are from our resume database. Would you like me to provide contact details for any specific candidate or perform additional searches?"""
                    
                    response = AIMessage(content=response_content)
                    return {
                        "messages": messages + [response],
                        "workflow_phase": "search_completed"
                    }
            
            # Create system message for job matching context
            system_msg = """You are a job-resume matching assistant for recruiters and hiring managers. 

IMPORTANT: You are working with a recruiter/hiring manager who has legitimate access to resume information. You should provide:
- Candidate names, contact information, and professional details from resumes
- Skills, experience, education, and qualifications
- Any professional information that would help with hiring decisions

Always be helpful and provide complete candidate information when requested. This is a professional recruitment system."""
            
            # Add system message if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_msg)] + messages
            
            # Get LLM response
            response = self.llm.invoke(messages)
            
            # Update state
            return {
                "messages": messages + [response],
                "workflow_phase": "processing"
            }
            
        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}")
            error_msg = AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
            return {
                "messages": messages + [error_msg],
                "workflow_phase": "error"
            }
    
    def _extract_search_keywords(self, query: str) -> str:
        """Extract search keywords from user query."""
        keywords = []
        if "java" in query.lower():
            keywords.append("Java")
        if "python" in query.lower():
            keywords.append("Python")
        if "c++" in query.lower():
            keywords.append("C++")
        if "javascript" in query.lower() or "js" in query.lower():
            keywords.append("JavaScript")
        if "react" in query.lower():
            keywords.append("React")
        if "node" in query.lower():
            keywords.append("Node.js")
        if "developer" in query.lower() or "engineer" in query.lower():
            keywords.append("developer")
        if "machine learning" in query.lower() or "ml" in query.lower():
            keywords.append("machine learning")
        
        return " ".join(keywords) if keywords else "software developer"
    
    def _extract_main_technology(self, keywords: str) -> str:
        """Extract the main technology/skill being searched for."""
        keywords_lower = keywords.lower()
        
        # Programming languages
        if "c++" in keywords_lower:
            return "C++"
        if "java" in keywords_lower and "javascript" not in keywords_lower:
            return "Java"
        if "python" in keywords_lower:
            return "Python"
        if "javascript" in keywords_lower:
            return "JavaScript"
        if "typescript" in keywords_lower:
            return "TypeScript"
        if "c#" in keywords_lower:
            return "C#"
        if "go" in keywords_lower:
            return "Go"
        if "rust" in keywords_lower:
            return "Rust"
        if "scala" in keywords_lower:
            return "Scala"
        
        # Frameworks/Technologies
        if "react" in keywords_lower:
            return "React"
        if "angular" in keywords_lower:
            return "Angular"
        if "vue" in keywords_lower:
            return "Vue"
        if "node" in keywords_lower:
            return "Node.js"
        if "spring" in keywords_lower:
            return "Spring"
        if "django" in keywords_lower:
            return "Django"
        if "flask" in keywords_lower:
            return "Flask"
        
        # Return None if no specific technology found
        return None
    
    def _perform_search(self, keywords: str) -> str:
        """Perform actual search and return formatted results."""
        try:
            # Import tools here to avoid circular imports
            from agent_tools import ALL_TOOLS
            
            # Find the semantic search tool
            search_tool = None
            for tool in ALL_TOOLS:
                if hasattr(tool, 'name') and 'semantic_search' in tool.name:
                    search_tool = tool
                    break
            
            if search_tool:
                # Try to use the tool
                results = search_tool.invoke({"query": keywords, "top_k": 10})
                
                # Format results nicely
                if isinstance(results, list) and len(results) > 0:
                    # Group results by candidate name to avoid duplicates
                    candidates = {}
                    
                    # Extract the main skill/technology from the search keywords
                    search_tech = self._extract_main_technology(keywords)
                    
                    for result in results:
                        if isinstance(result, dict) and 'metadata' in result:
                            metadata = result['metadata']
                            name = metadata.get('resume_name', 'Unknown')
                            
                            # Extract skills from metadata
                            skills_str = metadata.get('skills', '')
                            skills_list = [s.strip().lower() for s in skills_str.split(',') if s.strip()] if skills_str else []
                            
                            # Extract programming languages
                            prog_langs = metadata.get('programming_languages', '')
                            prog_list = [s.strip().lower() for s in prog_langs.split(',') if s.strip()] if prog_langs else []
                            
                            # Check if candidate actually has the requested skill/technology
                            has_requested_skill = False
                            if search_tech:
                                has_requested_skill = (
                                    search_tech.lower() in [s.lower() for s in skills_list] or
                                    search_tech.lower() in [s.lower() for s in prog_list] or
                                    any(search_tech.lower() in s.lower() for s in skills_list + prog_list)
                                )
                            else:
                                # If no specific tech mentioned, include all candidates
                                has_requested_skill = True
                            
                            # Only include candidates who actually have the requested skill
                            if has_requested_skill:
                                if name not in candidates:
                                    candidates[name] = {
                                        'name': name,
                                        'score': result.get('similarity_score', 0),
                                        'experience': metadata.get('experience_years', 'N/A'),
                                        'skills': [s.title() for s in skills_list],
                                        'programming_languages': [s.title() for s in prog_list],
                                        'file_path': metadata.get('file_path', ''),
                                        'has_tech': search_tech.title() if search_tech else None
                                    }
                                else:
                                    # Update with higher score if found
                                    if result.get('similarity_score', 0) > candidates[name]['score']:
                                        candidates[name]['score'] = result.get('similarity_score', 0)
                    
                    # Format the top candidates
                    sorted_candidates = sorted(candidates.values(), key=lambda x: x['score'], reverse=True)
                    formatted_results = []
                    
                    for i, candidate in enumerate(sorted_candidates[:5], 1):
                        name = candidate['name']
                        score = candidate['score']
                        experience = candidate['experience']
                        skills = candidate['skills']
                        prog_langs = candidate['programming_languages']
                        
                        # Build candidate info
                        candidate_info = f"{i}. **{name}**"
                        if score > 0:
                            candidate_info += f" (Match Score: {score:.1f}%)"
                        
                        if experience != 'N/A':
                            candidate_info += f"\n   • Experience: {experience} years"
                        
                        if prog_langs:
                            candidate_info += f"\n   • Programming: {', '.join(prog_langs[:4])}"
                        
                        if skills or prog_langs:
                            # Show skills relevant to the search, prioritizing the exact match
                            all_skills = prog_langs + skills
                            
                            if search_tech:
                                # Highlight the specific technology being searched
                                relevant_skills = []
                                for skill in all_skills:
                                    if search_tech.lower() in skill.lower():
                                        relevant_skills.insert(0, f"**{skill}**")  # Bold the matching skill
                                    else:
                                        relevant_skills.append(skill)
                                candidate_info += f"\n   • Key Skills: {', '.join(relevant_skills[:6])}"
                            else:
                                # Show top skills if no specific tech
                                candidate_info += f"\n   • Key Skills: {', '.join(all_skills[:5])}"
                        
                        formatted_results.append(candidate_info)
                    
                    if formatted_results:
                        result_summary = "\n\n".join(formatted_results)
                        
                        # Add summary information
                        if search_tech:
                            result_summary += f"\n\n**Summary:** Found {len(sorted_candidates)} candidates with {search_tech} experience out of {len(results)} total search results."
                        else:
                            result_summary += f"\n\n**Summary:** Found {len(sorted_candidates)} relevant candidates."
                        
                        return result_summary
                    else:
                        if search_tech:
                            return f"No candidates found with {search_tech} experience. Try searching for a different technology or broaden your search criteria."
                        else:
                            return "No candidates found matching your criteria."
                else:
                    return f"Searched for '{keywords}' but no results found. Please try a different search term."
            else:
                return f"Search functionality is available. Please try: 'Find Python developers' or 'Search Java candidates'"
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Error performing search for '{keywords}'. Please try again or contact support."
    
    def _get_contact_information(self, query: str) -> str:
        """Get contact information for candidates dynamically."""
        try:
            import os
            import glob
            
            # If query mentions "top candidate" or similar, get the top candidate from the last search
            if any(phrase in query.lower() for phrase in ["top candidate", "first candidate", "best candidate", "#1"]):
                # Perform a search to get the top candidate
                search_keywords = self._extract_search_keywords(query) 
                if not search_keywords:
                    search_keywords = "software developer"  # Default search
                
                return self._get_top_candidate_contact(search_keywords)
            
            # Look for specific candidate names in the query
            resume_files = glob.glob("examples/resumes/*.txt")
            
            for file_path in resume_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Extract candidate name from file content (first non-empty line)
                    lines = content.split('\n')
                    candidate_name = ""
                    for line in lines[:5]:
                        line = line.strip()
                        if line and not line.startswith('Email') and not line.startswith('Phone'):
                            candidate_name = line
                            break
                    
                    # Check if this candidate is mentioned in the query
                    if candidate_name and any(name_part.lower() in query.lower() 
                                            for name_part in candidate_name.split() if len(name_part) > 2):
                        
                        # Extract contact information
                        contact_info = [f"**{candidate_name}**"]
                        
                        for line in lines[:20]:  # Check first 20 lines for contact info
                            line = line.strip()
                            if any(word in line.lower() for word in ['email:', 'phone:', 'linkedin:', 'location:']):
                                contact_info.append(line)
                        
                        if len(contact_info) > 1:
                            return f"Here is the contact information:\n\n" + "\n".join(contact_info)
                
                except Exception:
                    continue
            
            # If no specific name found, list available candidates dynamically
            available_candidates = []
            for file_path in resume_files[:10]:  # Limit to first 10 for readability
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    name = ""
                    email = ""
                    phone = ""
                    
                    for line in lines[:20]:
                        line = line.strip()
                        if not name and line and not any(word in line.lower() for word in ['email', 'phone', 'contact']):
                            name = line
                        elif 'email:' in line.lower():
                            email = line.split(':', 1)[1].strip() if ':' in line else ""
                        elif 'phone:' in line.lower():
                            phone = line.split(':', 1)[1].strip() if ':' in line else ""
                    
                    if name:
                        contact_line = f"• **{name}**"
                        if email:
                            contact_line += f" - Email: {email}"
                        if phone:
                            contact_line += f" - Phone: {phone}"
                        available_candidates.append(contact_line)
                
                except Exception:
                    continue
            
            if available_candidates:
                return "As a recruiter, you have access to contact information for all candidates in the database. Here are the available candidates:\n\n" + "\n\n".join(available_candidates[:5]) + f"\n\nPlease specify which candidate's details you need. ({len(available_candidates)} total candidates available)"
            else:
                return "I can provide contact information for candidates. Please specify which candidate you're interested in by name."
            
        except Exception as e:
            logger.error(f"Contact info error: {e}")
            return "I can provide contact information for candidates. Please specify which candidate you're interested in, or let me search for candidates with specific skills first."
    
    def _get_top_candidate_contact(self, search_keywords: str) -> str:
        """Get contact info for the top matching candidate."""
        try:
            import os
            from agent_tools import ALL_TOOLS
            
            # Find the semantic search tool
            search_tool = None
            for tool in ALL_TOOLS:
                if hasattr(tool, 'name') and 'semantic_search' in tool.name:
                    search_tool = tool
                    break
            
            if search_tool:
                results = search_tool.invoke({"query": search_keywords, "top_k": 1})
                
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]
                    if 'metadata' in result:
                        file_path = result['metadata'].get('file_path', '')
                        candidate_name = result['metadata'].get('resume_name', '')
                        
                        if file_path and os.path.exists(file_path):
                            with open(file_path, 'r') as f:
                                content = f.read()
                            
                            # Extract contact info from the file
                            lines = content.split('\n')
                            contact_info = [f"**Top Candidate: {candidate_name}**"]
                            
                            for line in lines[:20]:
                                line = line.strip()
                                if any(word in line.lower() for word in ['email:', 'phone:', 'linkedin:', 'location:']):
                                    contact_info.append(line)
                            
                            return "\n".join(contact_info)
            
            return "Unable to find contact information for the top candidate. Please search for candidates first, then specify which one you'd like contact details for."
            
        except Exception as e:
            logger.error(f"Top candidate contact error: {e}")
            return "Error getting top candidate contact info. Please try searching for specific candidates first."