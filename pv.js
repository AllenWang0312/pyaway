/**
 * Created by wangpengcheng on 2019-12-23.
 */
if (typeof(fn_pvhitimgview) == "undefined") {
    var fn_pvhitimgview = true;
    FN_PV_JS_OBJ = {
        js_dom: "js.fengniao.com", imgsrc: "", pv_rport: function (a, c) {
            var b = a.indexOf(c);
            if (b > 0) {
                return a.substring(0, b)
            }
            if (b == 0) {
                return a.substring(1)
            }
            return a
        }, getRefUrl: function (b) {
            if (b.indexOf("ref0") > -1) {
                var a = /(?:\&|\?)ref0=([\s\S]*?)$/i;
                b = b.match(a);
                b = encodeURI(b[1]);
                return b
            }
        }, getDomain: function () {
            var a = "";
            hn = location.hostname;
            str = hn.replace(/\.(com|net|org|cn$)\.?.*/, "");
            if (str.lastIndexOf(".") == -1) {
                a = "." + hn
            } else {
                str = str.substring(str.lastIndexOf("."));
                a = hn.substring(hn.lastIndexOf(str))
            }
            return a
        }, getflash: function () {
            var b, a;
            if (window.ActiveXObject) {
                for (b = 12; b > 0; b--) {
                    try {
                        a = new ActiveXObject("ShockwaveFlash.ShockwaveFlash." + b);
                        return b + ".0"
                    } catch (c) {
                    }
                }
            } else {
                if (navigator.plugins && navigator.plugins.length) {
                    for (b = 0; b < navigator.plugins.length; b++) {
                        if (navigator.plugins[b].name.indexOf("Shockwave Flash") != -1) {
                            return navigator.plugins[b].description.split(" ")[2]
                        }
                    }
                }
            }
            return "Not enabled"
        }, readck: function (a) {
            var c = "";
            var b = a + "=";
            if (document.cookie.length > 0) {
                offset = document.cookie.indexOf(b);
                if (offset != -1) {
                    offset += b.length;
                    end = document.cookie.indexOf(";", offset);
                    if (end == -1) {
                        end = document.cookie.length
                    }
                    c = unescape(document.cookie.substring(offset, end))
                }
            }
            return c
        }, writeck: function (d, e, a) {
            var c = "";
            var b = this.getDomain();
            if (a != null) {
                c = new Date((new Date()).getTime() + a * 3600000);
                c = "; expires=" + c.toGMTString()
            }
            document.cookie = d + "=" + escape(e) + c + ";domain=" + b + ";path=/; "
        }, randck: function () {
            return Math.floor(Math.random() * 256)
        }, gettitle: function () {
            var b;
            if (typeof(encodeURIComponent) == "function") {
                if (document.title) {
                    if (window.RegExp) {
                        var a = new RegExp("^" + window.location.protocol + "//" + window.location.hostname + "\\s-\\s");
                        b = document.title.replace(a, "")
                    }
                } else {
                    b = document.title
                }
                b = encodeURIComponent(b)
            } else {
                b = ""
            }
            return b
        }, getRefer: function () {
            return document.referrer
        }, getDomainOf: function (a) {
            var b = a.split("/");
            hn = b[2];
            if (typeof(hn) == "undefined" || hn == "") {
                return
            }
            str = hn.replace(/\.(com|net|org|cn$)\.?.*/, "");
            if (str.lastIndexOf(".") == -1) {
                dm = "." + hn
            } else {
                str = str.substring(str.lastIndexOf("."));
                dm = hn.substring(hn.lastIndexOf(str))
            }
            return dm
        }, deleteck: function (c) {
            var b = "";
            var a = this.getDomain();
            b = new Date((new Date()).getTime() - 3600);
            b = "; expires=" + b.toGMTString();
            var d = this.readck(c);
            document.cookie = c + "=" + d + b + ";domain=" + a + ";path=/; "
        }, pv_d: function () {
            var k = window.location.protocol;
            var k = "https:";
            var b = new Date().getTime();
            var c = escape(b * 1000 + Math.round(Math.random() * 1000));
            var e = this.getDomain();
            if (e == ".fengniao.com") {
                var f = this.readck("bbuserid")
            }
            var a = location.href;
            if (a.indexOf("#") != -1) {
                a = a.substr(0, a.indexOf("#"))
            }
            if (a.indexOf("?") != -1) {
                a = a.substr(0, a.indexOf("?"))
            }
            if (top.location == self.location) {
                var g = this.getRefer();
                if (g.indexOf("#") != -1) {
                    g = g.substr(0, g.indexOf("#"))
                }
                if (g.indexOf("?") != -1) {
                    g = g.substr(0, g.indexOf("?"))
                }
                g = g.replace(/\</g, "");
                g = g.replace(/\>/g, "");
                if (a.indexOf(k + "//m.fengniao.com/") >= 0 || a.indexOf(k + "//m.2.fengniao.com/") >= 0 || a.indexOf(k + "//apis.fengniao.com/") >= 0 || a.indexOf(k + "//api.fengniao.com/") >= 0) {
                    this.imgsrc = k + "//wappv.fengniao.com/images/wap3g.gif?t=" + c + "&vuserid=" + f + "&url=" + a + "&referer=" + g
                } else {
                    this.imgsrc = k + "//pv.fengniao.com/images/pvhit0001.gif?t=" + c + "&vuserid=" + f + "&url=" + a + "&referer=" + g
                }
            } else {
                var i = document.referrer + "";
                i = i.substr(7);
                i = this.pv_rport(i, "/");
                i = this.pv_rport(i, ":");
                var g = this.getRefUrl(document.referrer);
                g = (g) ? g : document.referrer;
                if (g.indexOf("#") != -1) {
                    g = g.substr(0, g.indexOf("#"))
                }
                if (g.indexOf("?") != -1) {
                    g = g.substr(0, g.indexOf("?"))
                }
                g = g.replace(/\</g, "");
                g = g.replace(/\>/g, "");
                if (i.substr(i.length - 10) == "google.com" || i.substr(i.length - 9) == "baidu.com" || i.substr(i.length - 6) == "so.com" || i.substr(i.length - 6) == "360.cn") {
                    this.imgsrc = k + "//pv.fengniao.com/images/pvhit0001.gif?t=" + +c + "&vuserid=" + f + "&url=" + a + "&referer=" + g
                } else {
                    if (i.substr(i.length - 10) == "xiyuit.com") {
                        this.imgsrc = k + "//pv.fengniao.com/images/pvhit0002.gif?t=" + c + "&vuserid=" + f + "&url=" + a + "&referer=" + g
                    } else {
                        this.imgsrc = k + "//pv.fengniao.com/images/pvhit0003.gif?t=" + c + "&vuserid=" + f + "&url=" + a + "&referer=" + g
                    }
                }
            }
            if (this.imgsrc != "") {
                var j = this.readck("ip_ck");
                var d = this.js_dom;
                if (null == document.getElementById("pv_d")) {
                    document.write("<scr" + 'ipt type="text/javascript" id="pv_d" src="' + k + "//" + d + "/p.ht?h=&t=" + parseInt(c / 1000) + "&c=" + j + '"></scr' + "ipt>")
                } else {
                    var h = document.createElement("script");
                    h.src = window.location.protocol + "//" + d + "/p.ht?h=6&t=" + parseInt(c / 1000) + "&c=" + j;
                    h.type = "text/javascript";
                    document.getElementsByTagName("head")[0].appendChild(h)
                }
            }
        }
    };
    FN_PV_JS_OBJ.pv_d()
}
;