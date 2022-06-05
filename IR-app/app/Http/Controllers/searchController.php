<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class searchController extends Controller
{
    function get_res(Request $request)
    {
        $url = 'http://127.0.0.1:5000/my-route?q='.$request->input('search');
        $response = Http::get($url);
        // dd($url,$response->json());
        $res_all = $response->json();
        // dd($res_all);
        $results = $res_all[0];
        $corriction = $res_all[1];
        return view('show_res',['results'=>$results,'corr'=>$corriction]);
    }

    function show_eval()
    {
        $url = 'http://127.0.0.1:5000/my-route/evaluating';
        $response = Http::get($url);
        $evaluated_value = $response->json();
        $MMR = $evaluated_value[0];
        $precision = $evaluated_value[1];
        $recall = $evaluated_value[2];
        return view('show-eval',['MMR'=> $MMR, 'precision'=> $precision , 'recall'=> $recall]);
    }
}
