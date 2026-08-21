Fitted refusal directions from `refusal_direction.py --stage fit`.

Committed rather than kept in `do_not_track/` because they contain no generated text -- only a
normalized float16 vector per locus plus its fit statistics -- and because shipping them in the
image means a fresh Modal workspace needs no upload step to run the census.

    {run}_direction.pt  ->  {"directions": {locus: tensor}, "report": {locus: {...}}, ...}
