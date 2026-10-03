ObjC.import('AppKit');
function run(argv) {
  var pb = $.NSPasteboard.generalPasteboard;
  var out = {files: [], image: null, text: null};
  var opts = $({NSPasteboardURLReadingFileURLsOnlyKey: $.NSNumber.numberWithBool(true)});
  var urls = pb.readObjectsForClassesOptions($([$.NSURL]), opts);
  if (urls && urls.count > 0) {
    for (var i = 0; i < urls.count; i++) out.files.push(ObjC.unwrap(urls.objectAtIndex(i).path));
  }
  var png = pb.dataForType($.NSPasteboardTypePNG);
  if (!png || png.length == 0) {
    var tiff = pb.dataForType($.NSPasteboardTypeTIFF);
    if (tiff && tiff.length > 0) {
      var rep = $.NSBitmapImageRep.imageRepWithData(tiff);
      png = rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $({}));
    }
  }
  if (png && png.length > 0 && out.files.length == 0) {
    png.writeToFileAtomically(argv[0], true);
    out.image = argv[0];
  }
  var t = pb.stringForType($.NSPasteboardTypeString);
  if (t) out.text = ObjC.unwrap(t);
  return JSON.stringify(out);
}
