import os
from langchain_cohere import ChatCohere
from crewai import Crew, Process, Pipeline, Agent
from crewai import Task
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)
from crewai_tools import FileWriterTool

os.environ["SERPER_API_KEY"] = os.environ.get('SERPER_API_KEY') 

llm = ChatCohere(api_key=os.environ.get("Cohere_API_Key"), model = 'command-r-plus'  ,temperature=0)

search_tool = SerperDevTool()
# web_rag_tool = WebsiteSearchTool()  Can use the tool for deeper analysis and with more context consideration

Company_research_agent = Agent(
    role="Company and Industry Researcher",
    goal="Provide both comprehensive insights into the provided Company {question}  and it's Industry, identify its key offerings, and understand strategic focus areas of provided question {question}?",
    backstory="An expert analyst with a keen eye in market study, accurate insights about company and industry, to empower companies with the knowledge needed to thrive and stay competitive in their Industry ",
    tools = [search_tool],
    llm = llm,
    verbose=True,
)

Industry_task = Task(
    description='Use a Web browser tool and understand the industry and segment the company is working in (e.g., Automotive, Manufacturing, Finance, Retail, Healthcare, etc.) and analyze it',
    expected_output=' A summary of the industry including key characteristics, size, growth trends, and major players.',
    agent=Company_research_agent,
    async_execution=True,
    # tools=[search_tool],
    verbose=True,
)

Company_task = Task(
    description='Use a Web browser tool and understand Identify the company’s key offerings and strategic focus areas and analyze it',
    expected_output='A summary of the company including its vision, products, history, size, and global presence. Detailed list of products and services offered by the company.',
    agent=Company_research_agent,
    async_execution=True,
    # tools=[search_tool],
    verbose=True,
)

write_task = Task(
    description="Provide clear and simple representation of the context recieved",
    expected_output='Structured Information about the Company and Industry',
    agent=Company_research_agent,
    context=[Industry_task ,Company_task]
)

# Define your crews for Background Information
Background_context_crew = Crew(
    agents=[Company_research_agent],
    tasks=[Company_task, Industry_task, write_task],
    process=Process.sequential,
    verbose=True,
)

# Market Research and Use case generation crew

Market_research_agent = Agent(
    role="Conduct Market Research",
    goal="To provide a comprehensive understanding of current market dynamics, competitor activities, regulatory standards, and emerging technologies that can influence the company's strategic decisions in AI, ML, and automation at {question}?",
    backstory="The Market Research Agent acts as an analytical researcher focused on gathering and synthesizing information about industry trends and standards related to AI, ML, and automation within the company's sector ",
    tools = [search_tool],
    context = [write_task],
    llm = llm,
    verbose=True,
)

Market_Research_task = Task(
    description='''Utilize the search tool to collect up-to-date data on AI, ML, and automation trends specific to the company\'s industry.
                   Identify key competitors and analyze their adoption and implementation of AI, ML, and automation technologies.
                   ''',
    expected_output='Compile a detailed report summarizing findings, highlighting opportunities and threats within the market along with sources with clickable links.',
    agent=Market_research_agent,
    context = [write_task],
    # tools=[Websearch_tool],
    verbose=True,
)

Use_Case_generation_agent = Agent(
    role="Use Case Generation Agent",
    goal="To develop actionable use cases that leverage advanced AI technologies to improve company processes, enhance customer satisfaction, and boost operational efficiency at {question}?",
    backstory="The Use Case Generation Agent serves as an innovative strategist, responsible for conceptualizing and proposing practical applications of GenAI, LLMs, and ML technologies to address the company's specific needs.",
    tools = [search_tool],
    context = [Market_Research_task, write_task],
    llm = llm,
    process=Process.sequential,
    verbose=True,
)

Use_case_generate_task = Task(
    description='''Analyze the insights and data provided by the Market Research Agent.
                   Identify areas within the company's operations that can benefit from AI, ML, GenAI, and LLM technologies.
                   Generate a list of relevant and feasible use cases with detailed descriptions of potential impacts, benefits and their respective resource links.
                   Prioritize use cases based on alignment with company goals, potential ROI, feasibility, and resource requirements
                   ''',
    expected_output='Compile a detailed report summarizing findings, highlighting opportunities and threats within the market along with sources with clickable links.',
    agent=Use_Case_generation_agent,
    context=[Market_Research_task, write_task],
    # tools=[Websearch_tool],
    verbose=True,
)

# Initialize the tool
file_writer_tool = FileWriterTool()

structured_writer = Task(
    description="Present recommendations in a structured format to assist decision-makers in evaluating and selecting initiatives for implementation along with clickable resource links in a file",
    expected_output='File saved successfully',
    agent=Use_Case_generation_agent,
    context=[Market_Research_task ,Use_case_generate_task],
    tools=file_writer_tool,
    output_file='outputs/ai_usecase_summary.txt',
    create_directory=True
)

Marker_Research_Use_case_Gen_crew = Crew(
    agents=[Market_research_agent, Use_Case_generation_agent],
    tasks=[Market_Research_task, Use_case_generate_task, structured_writer],
    process=Process.sequential,
    verbose=True,
)


# Resource search and collection tools

Resource_Collect_Agent = Agent(
    role="Resource Asset Collection Agent",
    goal=" To facilitate the implementation of proposed use cases by providing a comprehensive collection of relevant resources from platforms like Kaggle, HuggingFace, and GitHub for {question}?",
    backstory="The Resource Asset Collection Agent serves as a data and resource curator, responsible for sourcing and compiling relevant datasets, models, and code repositories that align with the use cases generated by the Use Case Generation Agent.",
    tools = [search_tool],
    context = [structured_writer],
    llm = llm,
    process=Process.sequential,
    verbose=True,
)

Kaggle_task = Task(
    description='''Search kaggle for relevant resources and datasets for each use case
                   ''',
    expected_output='Gathered URLs and brief descriptions of each relevant resource.',
    agent=Resource_Collect_Agent,
    process=Process.sequential,
    context = [structured_writer],
    # tools=[Websearch_tool],
    verbose=True,
)

Github_task = Task(
    description='''Search github for relevant resources, repositories, projects and datasets for each use case
                   ''',
    expected_output='Gathered URLs and brief descriptions of each relevant resource.',
    agent=Resource_Collect_Agent,
    process=Process.sequential,
    context = [structured_writer],
    # tools=[Websearch_tool],
    verbose=True,
)

Huggingface_task = Task(
    description='''Search Huggingface for relevant resources, repositories, projects, blogs and datasets for each use case
                   ''',
    expected_output='Gathered URLs and brief descriptions of each relevant resource.',
    agent=Resource_Collect_Agent,
    process=Process.sequential,
    context = [structured_writer],
    # tools=[Websearch_tool],
    verbose=True,
)

resource_writer = Task(
    description="Present resources gathered in a structured format. Categorize the resources based on the specific use cases they support for easy navigation along with clickable resource links in a file",
    expected_output='File saved successfully',
    agent=Use_Case_generation_agent,
    context=[ Kaggle_task, Github_task, Huggingface_task],
    tools=file_writer_tool,
    process=Process.sequential,
    output_file='outputs/ai_resourses.txt',
    create_directory=True
)

Resource_Gen_crew = Crew(
    agents=[Company_research_agent, Market_research_agent, Use_Case_generation_agent, Market_research_agent, Use_Case_generation_agent],
    tasks=[Company_task, Industry_task, write_task, Market_Research_task, Use_case_generate_task, structured_writer, Huggingface_task, Github_task, Kaggle_task, resource_writer],
    process=Process.sequential,
    verbose=True,
)



# Use Websearch tool for deeper analysis

# Websearch_tool = WebsiteSearchTool(
#     config=dict(
#         llm=dict(
#             provider="cohere", # or google, openai, anthropic, llama2, ...
#             config=dict(
#                 model="command-r-plus",
#                 # temperature=0.5,
#                 # top_p=1,
#                 # stream=true,
#             ),
#         ),
#         embedder=dict(
#             provider="cohere", # or openai, ollama, ...
#             config=dict(
#                 model="embed-english-v3.0",
#                 # task_type="retrieval_document",
#                 # title="Embeddings",
#             ),
#         ),
#     )
# )