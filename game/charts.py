import matplotlib.pyplot as pylot

def genera_pie_chart():
    labels = ['A', 'B', 'C']
    values = [200, 34, 120]
    
    fig, ax =pylot.subplots()
    ax.pie(values, labels=labels)
    pylot.savefig('pie.png')
    pylot.close()