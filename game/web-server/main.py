import store
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def get_list():
    return[1,2,3,]

@app.get('/contact')
def get_list():
    return{name: 'Platzi'}


def run():
    store.get_categories()
    

if __name__ == '__main__':
    run()