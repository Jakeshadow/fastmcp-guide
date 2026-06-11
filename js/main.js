(function () {
  'use strict';

  // === Mobile nav toggle ===
  var toggle = document.querySelector('.nav-toggle');
  var navLinks = document.querySelector('.nav-links');

  if (toggle && navLinks) {
    toggle.addEventListener('click', function () {
      navLinks.classList.toggle('open');
    });

    // Close mobile nav on link click
    navLinks.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        navLinks.classList.remove('open');
      });
    });
  }

  // === Active nav link on scroll ===
  var sections = document.querySelectorAll('section[id]');
  var navAnchors = document.querySelectorAll('.nav-links a');
  var sectionTops = [];

  function cacheOffsets() {
    sectionTops = [];
    sections.forEach(function (s) {
      sectionTops.push({ id: s.getAttribute('id'), top: s.offsetTop });
    });
  }

  function updateActiveLink() {
    var scrollPos = window.scrollY + 80;
    var current = '';

    for (var i = 0; i < sectionTops.length; i++) {
      if (scrollPos >= sectionTops[i].top) {
        current = sectionTops[i].id;
      }
    }

    navAnchors.forEach(function (a) {
      a.classList.remove('active');
      if (a.getAttribute('href') === '#' + current) {
        a.classList.add('active');
      }
    });
  }

  cacheOffsets();
  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(cacheOffsets, 100);
  }, { passive: true });
  window.addEventListener('scroll', updateActiveLink, { passive: true });
  updateActiveLink();
})();
