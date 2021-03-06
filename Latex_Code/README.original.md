
What is that?
============  

This thesis latex template provides a skeleton for a diploma thesis with a corresponding make file.

How to get Started
==============

First you should be in diploma. Tex all occurrences of Otto
Replace the pattern man, your title, and your caregiver.

`diploma.tex` wants to embed your task as a PDF. It is looking `Images/Diplom-aufgabe. pdf`, which must be an A4 page. With

    Convert <dein-gescanntes-bild>Diplom-aufgabe. pdf

You should be able to convert any image file into a PDF if ImageMagick is installed on your system.

At this point `make` should produce a `diploma.pdf`.

The template supports both English and German text. English is Set by default. For German text The last `selectlanguage` can be Simply omit the call to `diploma.tex`.


Embed graphics
==================

Graphics should be placed in the ' images/' directory and
Makefile entered in the corresponding ' DOC_IMG_ * ' variable
Be. Currently, graphics in the formats PDF, PNG and JPEG are
Supported. PDF is suitable for vector graphics and can be used by
Generated by most vector graphics programs (Inkscape, OpenOffice
Draw,...).

Tipps
=====

This file currently contains a collection of tips and tricks, as well as
Some background information.

- passive voice: **do not use it**
  - There is a Makefile template checking for 'Bugs in Writing' according
    to the book of the same name (`make checkbiw`). Diction must be installed
    somewhere in the path (check out the directory
    `checkbiw/diction` for details).
  - Vim users can add detection for passive voice and *weasel words* via
    Björn's [`vim-weasel` package](https://github.com/bjoernd/vim-weasel)
- font sizes in images: adapt to other text size
   (ideally, use PGF/TikZ and PGFPlots)
- avoid missing meta data in PDF files (title, keywords, author)
- "good" title page
- use biblatex for references, it pays off fast
- convert images to correct include types (vector formats, e.g. PDF)
- protected spaces between, e.g., `Figure~1`, `~\cite{xyz}`
- units: use the `units` package to typeset units
- French spacing: tell latex what is an end of sentence with `\@.`
  where it cannot know it (e.g., `This is a sentence ending on an
  abbreviation BASIC\@.  Next sentence.`)
  - Again, you can try to detect a good portion of French spacing
    using an automatic algorithm (`make check-french-spacing`).
    Improvements are welcome.
- listings with at least three elements have a
  comma before the last and/or (*serial* or *Oxford comma*):
  *"Set A contains elements a, b, and c."*
- more stylistic information can be found in *Bugs in writing* (BIW)
  by Lyn Dupré



Special tips from Frank
=========================

- I use in the template coma-script (' Scrbook '), which mainly
  Intended for the German-speaking world. Coma script can also
  be used internationally, but the format is
  English language works somewhat unusual.

- A note about creating the graphics: Many use xfig, I
  Created my graphics with OpenOffice draw. There you have more
  Possibilities. Simply export as PDF. So that all graphics
  Have an even size, I just always have the
  Set the page size so that the drawing is fully captured. Then
  I always set the same font size. When incorporating the
  Graphics in the Latex-file I have a factor, e.g.

    includegraphics [width = 190  Figurewidth] {architecture}

Built-in. I then simply put the factor in the beginning with

    setlength{figurewidth}{.070cm}

And I'm able to resize all the graphics at the same time.  
  The number 190 from the Includegraphics statement comes from the selected Page size in OpenOffice Draw (190mm).

- If you have a lot of graphics that you want to align exactly, a

    usepackage{placeins}

Useful. Then you can at the end of a page

FloatBarrier  
Write and force the output of all still open graphics
  This point. I'm aware that this is a little ugly, but
  Sometimes you really need to do this, for example, if all the measured values
  On one side.
