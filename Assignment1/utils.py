def get_year_or_country() -> bool:
    while(1):
        inp = input("Show data by country or by year (c/y): ")
        if(inp == "y"):
            return True
        if(inp == "c"):
            return False
        
def get_display_name() -> bool:
    while(1):
        inp = input("Do you want country names to be displayed on the chart (y/n): ")
        if(inp == "y"):
            return True
        if(inp == "n"):
            return False