<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet"
        integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
    <title>Evaluating</title>
</head>

<body>

    <nav class="navbar navbar-expand-lg bg-light">
        <div class="container-fluid">

            <a class="navbar-brand" href="{{ route('welcome') }}">DOC-IR</a>
            <div class="d-flex flex-row-reverse bd-highlight">

                <div class="collapse navbar-collapse p-2 bd-highlight" id="navbarSupportedContent">
                    <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                        <li class="nav-item">
                            <a class="nav-link active" aria-current="page" href="{{ route('welcome') }}">Home</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ route('evaluating') }}">Evaluating</a>
                        </li>
                    </ul>
                </div>

            </div>

        </div>
    </nav>



    <div class="container mt-5">
        <table class="table table-striped table-bordered ">
            <thead>
                <tr>
                    <th scope="col">#</th>
                    <th scope="col">Evl Name</th>
                    <th scope="col" class="d-flex justify-content-center">Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <th scope="row">1</th>
                    <td>MMR@10</td>
                    <td class="d-flex justify-content-center">{{ round($MMR, 2) * 100 }}%</td>
                </tr>
                <tr>
                    <th scope="row">2</th>
                    <td>Precision@10</td>
                    <td colspan="4">
                        <div class="row">
                            @foreach ($precision as $item)
                                <div class="col">
                                    {{ round($item, 2) * 100 }}%
                                </div>
                            @endforeach
                        </div>
                    </td>
                </tr>
                <tr>
                    <th scope="row">3</th>
                    <td>Recall</td>
                    <td colspan="4">
                        <div class="row">
                            @foreach ($recall as $item)
                                <div class="col">
                                    {{ round($item, 2) * 100 }}%
                                </div>
                            @endforeach
                        </div>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>


</body>

</html>
