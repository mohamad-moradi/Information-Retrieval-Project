$("#form").submit(function(e) {
    e.preventDefault()

    var request = new XMLHttpRequest()

    var query = $("#search").val()

    var api = 'http://127.0.0.1:5000/my-route?q='+query

    console.log(api)

    $.get(api,function(data) {
        console.log(data)
    })
})