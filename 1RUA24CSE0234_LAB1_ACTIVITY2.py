import requests
import csv
from bs4 import BeautifulSoup

def scrape_books_from_website():
    """
    Scrapes book title, price, and star rating from a fictional books website.
    This demonstrates collecting structured data from an unstructured HTML page.
    The script now scrapes data from all pages of the website.
    """
    print("--- Starting Web Scraping Mini-Project ---")

    base_url = "http://books.toscrape.com/catalogue/"
    next_page_url = "page-1.html"  # Start with the first page

    all_scraped_books = []
    page_count = 0

    while next_page_url:
        page_count += 1
        current_url = requests.compat.urljoin(base_url, next_page_url)
        print(f"Scraping data from: {current_url}")

        try:
            response = requests.get(current_url)
            response.raise_for_status()

            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            book_articles = soup.find_all('article', class_='product_pod')

            for book in book_articles:
                title = book.h3.a['title']
                price_str = book.find('p', class_='price_color').text
                price = float(price_str.replace('Â£', '').strip())
                rating_class = book.find('p', class_='star-rating')['class'][1]
                rating = rating_class

                all_scraped_books.append({
                    'title': title,
                    'price': price,
                    'rating': rating
                })

            # Check for next page
            next_button = soup.find('li', class_='next')
            if next_button:
                next_page_url = next_button.a['href']
            else:
                next_page_url = None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from website: {e}")
            return None
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return None

    print(f"\nSuccessfully scraped a total of {len(all_scraped_books)} books from {page_count} pages.")
    return all_scraped_books

def save_to_csv(data, filename):
    """
    Saves a list of dictionaries to a CSV file.
    """
    if not data:
        print(f"No data to save to {filename}. Skipping.")
        return

    keys = data[0].keys()

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

        print(f"Data successfully saved to {filename}")

    except IOError as e:
        print(f"Error saving data to CSV file: {e}")

if __name__ == "__main__":
    books_data = scrape_books_from_website()

    if books_data:
        save_to_csv(books_data, 'books_data.csv')

    print("\n--- Process Complete ---")
