ObjC.import('Vision');
ObjC.import('Foundation');
function run(argv) {
  var url = $.NSURL.fileURLWithPath(argv[0]);
  var handler = $.VNImageRequestHandler.alloc.initWithURLOptions(url, $({}));
  var req = $.VNRecognizeTextRequest.alloc.init;
  req.recognitionLevel = 0;
  req.usesLanguageCorrection = true;
  var ok = handler.performRequestsError($([req]), null);
  if (!ok) return '';
  var res = req.results, out = [];
  for (var i = 0; i < res.count; i++) {
    var cand = res.objectAtIndex(i).topCandidates(1);
    if (cand.count > 0) out.push(ObjC.unwrap(cand.objectAtIndex(0).string));
  }
  return out.join('\n');
}
