<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documents retrieval system</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet"
        integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
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

    <div class="container">

        <img src="img/logo.png" class="rounded mx-auto d-flex justify-content-center" alt="logo">
        <form action="{{ route('results') }}" id="form" autocomplete="off" method="POST">
            @csrf
            <div class="form-group">
                <input type="text" name="search" id="search" class="form-control" placeholder="search">
            </div>
            <div class="form-group d-flex justify-content-center m-2">
                <button class="btn btn-primary btn-block">Serach</button>
            </div>
        </form>


        @if ($corr != '-no change-')
            <p class="lead"> Did you mean : <a href="#"
                    class="link-danger text-decoration-none">{{ $corr }}</a> ?</p>
        @endif


        @foreach ($results as $key => $val)
            {{-- <p> {{ $item }} </p> --}}
            <div class="accordion m-3" id="accordionExample<?php echo $key; ?>">
                <div class="accordion-item">
                    <h2 class="accordion-header" id="headingOne<?php echo $key; ?>">
                        <button class="accordion-button" type="button" data-bs-toggle="collapse"
                            data-bs-target="#collapseOne<?php echo $key; ?>" aria-expanded="true"
                            aria-controls="collapseOne<?php echo $key; ?>">
                            {{ $key }}
                        </button>
                    </h2>
                    <div id="collapseOne<?php echo $key; ?>" class="accordion-collapse collapse show"
                        aria-labelledby="headingOne<?php echo $key; ?>"
                        data-bs-parent="#accordionExample<?php echo $key; ?>">
                        <div class="accordion-body">
                            <strong>{{ $val }}</strong>
                        </div>
                    </div>
                </div>
            </div>
        @endforeach
    </div>

</body>
<!-- JavaScript Bundle with Popper -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/js/bootstrap.bundle.min.js"
integrity="sha384-pprn3073KE6tl6bjs2QrFaJGz5/SUsLqktiwsUTF55Jfv3qYSDhgCecCxMW52nD2" crossorigin="anonymous"></script>

</html>
