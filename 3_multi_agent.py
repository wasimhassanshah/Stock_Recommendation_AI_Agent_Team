from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools

# Load environment variables
load_dotenv()

# Define the research agent
research_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use proper tabular format to display data."],
    
)

# Define the data analysis agent
data_analysis_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_fundamentals=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Analyze the data and provide insights."],
    
)

# Define the plotting agent (assuming a plotting tool is available)
plotting_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[],  # Add plotting tools here if available
    show_tool_calls=True,
    markdown=True,
    instructions=["Create visualizations based on the provided data."],
     
)

# Define the query interaction agent
query_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[],
    show_tool_calls=True,
    markdown=True,
    instructions=["Interact with user queries effectively."],
   
)

# Define the stock recommendation agent
recommendation_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Provide recommendations on where to buy and sell stocks."],
    
)

# Example usage of agents
research_agent.print_response("Summarize and compare analyst recommendations for Apple and Microsoft")
data_analysis_agent.print_response("Analyze the fundamentals of Apple and Microsoft")
plotting_agent.print_response("Plot the stock price trends for Apple and Microsoft")
query_agent.print_response("How can I invest in technology stocks?")
recommendation_agent.print_response("Where should I buy or sell Apple stocks?")
