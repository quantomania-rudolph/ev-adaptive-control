from playwright.sync_api import sync_playwright
f="file://" + __import__("os").path.abspath("dashboard/dashboard.html") + ""
res=[]
def chk(n,ok,i=""): res.append(ok); print(("PASS " if ok else "FAIL ")+n,i)
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport=dict(width=1100,height=800))
    errs=[]; pg.on("pageerror",lambda e:errs.append(str(e))); pg.on("console",lambda m:errs.append(m.text) if m.type=="error" else None)
    pg.goto(f); pg.wait_for_timeout(500)
    bad=lambda: pg.evaluate("""()=>/NaN|undefined|Infinity/.test(document.body.innerText)""")
    chk("initial no NaN/undefined text",not bad())
    for i in range(30):   # rapid scenario switching incl. while playing
        pg.click("#scenseg button:nth-child(%d)"%(i%3+1))
        if i%4==0: pg.click("#play")
    pg.wait_for_timeout(400); chk("30 rapid switches: no errors",not errs,errs[:3]); chk("no NaN after switching",not bad())
    pg.click("#scenseg button:nth-child(1)")
    # hide all runs but one, then try hiding the last
    for name in ["A baseline","C robust","D anti","Dg anti"]:
        pg.locator("#charts-run .lg",has_text=name).first.click()
    pg.locator("#charts-run .lg",has_text="B naive").first.click()   # last one: must stay visible
    n=pg.locator("#charts-run .lg[aria-pressed=true]").count(); chk("cannot hide the last run",n>=1,n)
    # scrub extremes
    pg.evaluate("document.getElementById('scrub').value=0;document.getElementById('scrub').dispatchEvent(new Event('input'))"); chk("scrub to 0 ok",not bad() and not errs)
    pg.evaluate("document.getElementById('scrub').value=1000;document.getElementById('scrub').dispatchEvent(new Event('input'))"); chk("scrub to end ok",not bad() and not errs)
    # click on chart sets time
    svg=pg.locator("#charts-run svg.plot").first; bb=svg.bounding_box(); pg.mouse.click(bb["x"]+bb["width"]*0.5,bb["y"]+60)
    tl=pg.inner_text("#timelabel"); chk("chart click moves time",not tl.startswith("8.00"),tl)
    # resize narrow & wide
    for w in (320,390,768,1600):
        pg.set_viewport_size(dict(width=w,height=800)); pg.wait_for_timeout(300)
        sw=pg.evaluate("document.documentElement.scrollWidth"); chk(f"no horizontal scroll at {w}px",sw<=w,sw)
    # every node card clickable
    pg.set_viewport_size(dict(width=1100,height=800))
    for nid in ["req","ctl","sat","mot","load","bat","bus","math"]:
        pg.click(f".node[data-id={nid}]"); rows=pg.locator("#detail tr").count(); chk(f"card {nid} detail rows",rows>1,rows)
    chk("no NaN in detail",not bad())
    # catalog filter, empty result
    pg.fill("#q","zzzzzz"); chk("filter no match -> 0",pg.inner_text("#qcount").lower().startswith("0")); pg.fill("#q","")
    chk("filter reset -> 184",pg.inner_text("#qcount").lower().startswith("184"))
    # keyboard: tab to play, press space
    pg.focus("#play"); pg.keyboard.press("Space"); pg.wait_for_timeout(300); chk("keyboard play works",pg.inner_text("#play") in("Pause","Resume"),pg.inner_text("#play"))
    # zoom toggle
    pg.click("text=Zoom to"); chk("zoom toggle no error",not errs)
    # all 3 scenarios, every chart svg exists or na
    for s in (1,2,3):
        pg.click("#scenseg button:nth-child(%d)"%s); pg.wait_for_timeout(300)
        n_svg=pg.locator("svg.plot").count(); n_na=pg.locator(".na").count(); chk(f"scenario {s}: charts+placeholders",n_svg+n_na==15,(n_svg,n_na))
    chk("total errors none",not errs,errs[:3])
    pg.emulate_media(reduced_motion="reduce"); pg.reload(); pg.wait_for_timeout(400); chk("reduced-motion loads",not errs)
    b.close()
print("FAILS",res.count(False))
